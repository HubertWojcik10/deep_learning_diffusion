import torch 
from tqdm import tqdm 
from model.unet import Unet 

def train(epochs_num:int, train_loader, noise_scheduler, lr:float=0.001):

    unet = Unet()
    device= 'cuda' if torch.cuda.is_available() else 'cpu'  
    optimizer= torch.optim.Adam(unet.parameters(), lr=lr)
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
            outputs = unet(noisy_inputs, t)
            loss = criterion(outputs, inputs_noise)

            #backpropagate
            loss.backward()
            optimizer.step()

            print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')