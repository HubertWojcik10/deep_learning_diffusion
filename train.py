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
import matplotlib.pyplot as plt
import torchvision
import os
from torchvision.utils import make_grid

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
    sample_save_dir: str = "samples",
    test_loader = DataLoader
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
        #evaluate on test set
        #eval_loss = evaluate_on_test_set(model, test_loader, noise_scheduler, device)
        #print(f'Epoch: {epoch}, Test Loss: {round(eval_loss, 5)}')
        # Evaluate FID at intervals
        if epoch % fid_eval_interval == 0:
            print("Generating images for FID computation...")
            xt = torch.randn((fid_sample_size, 1, 28, 28)).to(device)
            sampled_images = sample(model, xt, noise_scheduler, params, device, 100)

            # # Save sampled images for visualization
            # for idx, img in enumerate(sampled_images):
            #     save_path = os.path.join(sample_save_dir, f"epoch_{epoch}_sample_out_{idx}.png")
            #     torchvision.utils.save_image(img, save_path)

            print("Calculating FID...")
            inception_model = prepare_inception_model(device)
            fid_score = compute_fid(activations_path, sampled_images.cpu().numpy(), inception_model, device)
            print(f"FID at epoch {epoch}: {round(float(fid_score), 3)}")

        print(f'Epoch: {epoch}, Train Loss: {round(sum(train_loss)/len(train_loss), 5)}')
        
        if ".pth" in save_path:
            save_path = save_path.replace(".pth", "")

    torch.save(model.state_dict(), f"{save_path}_{curr_time}_e{epoch}.pth")
       

def sample(model, xt, scheduler, config, device, batch_size):
    """
    Sample stepwise by going backward one timestep at a time, processing in batches.
    """
    images_generated = []
    model.eval()

    with torch.no_grad():
        for i in tqdm(reversed(range(config['num_timesteps']))):
            # Process in batches
            for batch_start in range(0, xt.size(0), batch_size):
                batch_end = min(batch_start + batch_size, xt.size(0))
                xt_batch = xt[batch_start:batch_end]
                
                # Get prediction of noise
                noise_pred = model(xt_batch, torch.as_tensor(i).unsqueeze(0).to(device))
                xt_batch, _ = scheduler.sample_prev_timestep(xt_batch, noise_pred, torch.as_tensor(i).to(device))
                
                xt[batch_start:batch_end] = xt_batch  # Update corresponding batch

            if i == 0:
                ims = torch.clamp(xt, -1., 1.).detach().cpu()
                ims = (ims + 1) / 2  # Normalize back to [0, 1]
                images_generated.append(ims)

    ims_batch = images_generated[-1]

    resize = transforms.Resize((299, 299))
    ims_resized_batch = resize(ims_batch)

    # Convert grayscale to RGB by repeating the channels
    ims_rgb_batch = ims_resized_batch.repeat(1, 3, 1, 1)
    return ims_rgb_batch


def evaluate_on_test_set(model, test_loader, noise_scheduler, device):
    """
    Evaluate the model on a test set and return the average loss.
    """
    model.eval()  # Set model to evaluation mode
    criterion = torch.nn.MSELoss()
    test_loss = []

    with torch.no_grad():  # No need to compute gradients during evaluation
        for inputs, _ in tqdm(test_loader):
            inputs = inputs.float().to(device)

            # Sample timestep
            t = torch.randint(0, 1000, (inputs.shape[0],)).to(device)

            # Add noise to inputs
            noisy_inputs, inputs_noise = noise_scheduler.add_noise(inputs, t)

            # Forward pass
            outputs = model(noisy_inputs, t)
            loss = criterion(outputs, inputs_noise)
            test_loss.append(loss.item())

    average_test_loss = sum(test_loss) / len(test_loss)
    return average_test_loss


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    params = load_params("config.json")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, num_workers=2)

    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, num_workers=2)

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
        save_path=params["model_path"], 
        test_loader = test_loader
    )