import torch 
from tqdm import tqdm 
from model.unet import Unet 
from dataset.mnist_dataset import MnistDataset
from utils.noise_scheduler.scheduler import NoiseScheduler
from utils.load_params import load_params
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def train(model,device, epochs_num:int, train_loader, noise_scheduler, lr:float=0.0001):
  
    optimizer= torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs_num):    
        train_loss=[]
        val_loss=[]
        model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            inputs= inputs.float().to(device)
            
            #sample timestep
            t = torch.randint(0, 1000, (inputs.shape[0],)).to(device)
            #adding noise to inputs
            noisy_inputs, inputs_noise = noise_scheduler.add_noise(inputs, t)

            #forward pass
            outputs = model(noisy_inputs, t)
            loss = criterion(outputs, inputs_noise)
            train_loss.append(loss.item())

            #backpropagate
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss}')

        plot_loss(train_loss)
        print(f'Epoch: {epoch}, Train Loss: {sum(train_loss)/len(train_loss)}')
        
        torch.save(model.state_dict(), f'./model/model.pth')


def plot_loss(train_loss):
    plt.plot(train_loss, label='train')
    plt.legend()
    plt.show()
    plt.savefig('loss.png')


if __name__ == '__main__':
    params = load_params("config.json")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # train_size = int(0.8 * len(trainset))  # 80% for training
    # val_size = len(trainset) - train_size  # 20% for validation
    # train_set, val_set = random_split(trainset, [train_size, val_size])
    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, num_workers=2)
    #val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=True, num_workers=2)

    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, num_workers=2)

    model = Unet()
    scheduler = NoiseScheduler(device, params["num_timesteps"], params["beta_start"], params["beta_end"])

    train(model, device, params["epochs_num"], train_loader, scheduler, params["learning_rate"])