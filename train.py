from torchvision import datasets
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.fid import *
from utils.noise_scheduler.scheduler import NoiseScheduler
from utils.load_params import load_params
from model.unet import Unet
from datetime import datetime
import warnings

def train(
    model: torch.nn.Module,
    device: str,
    epochs_num: int,
    train_loader: DataLoader,
    noise_scheduler: NoiseScheduler,
    save_path: str,
    lr: float,
    activations_path: str, 
    fid_eval_interval: int=1,
    fid_sample_size: int = 100,
):
    """
        Train function for the diffusion model
        Train the model & compute the FID score
    """
    optimizer= torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    curr_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    # epoch loop
    for epoch in range(epochs_num):
        train_loss = []
        model.train()

        for i, (inputs, _) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            inputs= inputs.float().to(device)
            
            # sample timestep
            t = torch.randint(0, 1000, (inputs.shape[0],)).to(device)

            # adding noise to inputs
            noisy_inputs, inputs_noise = noise_scheduler.add_noise(inputs, t)

            # forward pass
            outputs = model(noisy_inputs, t)
            loss = criterion(outputs, inputs_noise)
            train_loss.append(loss.item())

            # backpropagate
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {i}, Loss: {round(float(loss), 5)}')      

        # FID score
        inception_model = prepare_inception_model(device)

        if epoch % fid_eval_interval == 0:
            fake_images = []

            for inputs, _ in train_loader:
                inputs = inputs.float().to(device)
                t = torch.randint(0, 1000, (inputs.shape[0],)).to(device)
                noisy_inputs, _ = noise_scheduler.add_noise(inputs, t)
                fake_images_batch = model(noisy_inputs, t).detach().cpu()
                fake_images.extend(fake_images_batch)

            fake_images = torch.stack(fake_images)  # converting list to tensor
            fake_images = fake_images.repeat(1, 3, 1, 1)  # converting to RGB

            fid_score = compute_fid(activations_path, fake_images, inception_model, device)
            print(f'FID at epoch {epoch}: {round(float(fid_score), 3)}')

            #TODO: should we save the fake images here?

        plot_loss(train_loss)
        print(f'Epoch: {epoch}, Train Loss: {round(sum(train_loss)/len(train_loss), 5)}')
        
        if ".pth" in save_path:
            save_path = save_path.replace(".pth", "")

        torch.save(model.state_dict(), f"{save_path}_{curr_time}_e{epoch}.pth")


def plot_loss(train_loss):
    """
        Plot the train loss and save
    """
    plt.plot(train_loss, label='train')
    plt.legend()
    plt.show()
    plt.savefig('loss.png')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    params = load_params("config.json")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, num_workers=2)

    # Just take first two batches for testing
    train_loader = iter(train_loader)
    train_loader = [next(train_loader), next(train_loader)]

    model = Unet()
    scheduler = NoiseScheduler(DEVICE, params["num_timesteps"], params["beta_start"], params["beta_end"])

    train(
        model=model,
        device=DEVICE,
        epochs_num=params["epochs_num"],
        train_loader=train_loader,
        noise_scheduler=scheduler,
        lr=params["learning_rate"],
        activations_path=params["activations_path"],
        save_path=params["model_path"]
    )