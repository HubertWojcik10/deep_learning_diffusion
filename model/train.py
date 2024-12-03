import torch 
from tqdm import tqdm 
from model.unet import Unet 
from utils.fid import prepare_inception_model, compute_fid

def train(epochs_num:int, train_loader, noise_scheduler, lr:float=0.001, fid_eval_interval:int=5, activations_real_path:str=","):

    unet = Unet()
    device= 'cuda' if torch.cuda.is_available() else 'cpu'  
    optimizer= torch.optim.Adam(unet.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    #FID score initialization
    inception_model = prepare_inception_model(device)
    activations_real = torch.load(activations_real_path)
    
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

        #getting the FID score
        if epoch % fid_eval_interval == 0:

            fake_images = []
            for inputs in train_loader:
                inputs = inputs.float().to(device)
                t = torch.randint(0, 1000, (inputs.shape[0],)).to(device)
                noisy_inputs, _ = noise_scheduler.add_noise(inputs, t)
                fake_images_batch = unet(noisy_inputs, t).detach().cpu()
                fake_images.extend(fake_images_batch)

            fake_images = torch.stack(fake_images)  #converting list to tensor

            fid_score = compute_fid(activations_real, fake_images, inception_model, device)
            print(f'FID at epoch {epoch}: {fid_score}')

            #can we save the fake images here?