import torch
import torchvision.transforms as transforms
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from fid import prepare_inception_model, get_activations

def get_real_activations(loader, model, device):
    """
    Compute activations for real images using the Inception v3 model.
    """
    activations_list = []

    for inputs, _ in tqdm(loader, desc="Processing Real Activations"):
        inputs = inputs.to(device)
        activations = get_activations(inputs, model, device)
        activations_list.append(activations)
        
    return torch.tensor(np.concatenate(activations_list, axis=0))

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loading the MNIST (same as train.py file)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    # v3 inception model
    inception_model = prepare_inception_model(device)

    # Computing the activations
    print("Computing real activations...")
    activations = get_real_activations(loader, inception_model, device)

    # Saving the activations
    torch.save(activations, './real_activations.pth')
    print("Real activations saved to './real_activations.pth'")
