import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
from scipy.linalg import sqrtm
from torch.cuda.amp import autocast

def prepare_inception_model(device):
    model = models.inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()  # Removing the classification layer
    model.eval()
    model.to(device)
    return model

def get_activations(images, model, device, batch_size=8):
    activations = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        batch = nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        with autocast():
            with torch.no_grad():
                batch = (batch - 0.5) / 0.5
                batch_activations = model(batch).cpu().detach()
        activations.append(batch_activations)
    return torch.cat(activations, dim=0)

def compute_fid(activations_real_path, generated_images, inception_model, device):
    activations_real = torch.load(activations_real_path)
    mu_real, sigma_real = activations_real.mean(axis=0), np.cov(activations_real, rowvar=False)
    generated_images_tensor = torch.tensor(generated_images)
    activations_fake = get_activations(generated_images_tensor, inception_model, device)
    mu_fake, sigma_fake = activations_fake.mean(axis=0), np.cov(activations_fake, rowvar=False)

    diff = mu_real - mu_fake
    covmean, _ = sqrtm(sigma_real @ sigma_fake, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid
