import torch 
from tqdm import tqdm 
from model.unet import Unet 
from dataset.mnist_dataset import MnistDataset
from utils.noise_scheduler import NoiseScheduler
from utils.load_params import load_params
from torch.utils.data import DataLoader

def train(model, epochs_num:int, train_loader, noise_scheduler, lr:float=0.001):

   
    device= 'cuda' if torch.cuda.is_available() else 'cpu'  
    optimizer= torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs_num):    
        
        for i, inputs in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            inputs= inputs.float().to(device)
            
            #sample timestep
            t = torch.randint(0, 1000, (inputs.shape[0],)).to(device)
            #adding noise to inputs
            noisy_inputs, inputs_noise = noise_scheduler.add_noise(inputs, t)

            #forward pass
            outputs = model(noisy_inputs, t)
            loss = criterion(outputs, inputs_noise)

            #backpropagate
            loss.backward()
            optimizer.step()

            print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')
            

if __name__ == '__main__':
    params = load_params("config.json")

    train_dataset = MnistDataset(root_dir='data', train=True)
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

    test_dataset = MnistDataset(root_dir='data', train=False)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

    model = Unet()
    model.train()
    scheduler = NoiseScheduler(params["num_timesteps"], params["beta_start"], params["beta_end"])

    train(model, params["epochs_num"], train_loader, scheduler, params["lr"])