import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
from scipy.linalg import sqrtm


def prepare_inception_model(device):
    """
    Loads the Inception v3 model for FID calculation.
    Removes the final classification layer and sets to evaluation mode.
    """
    model = models.inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()  # Removing the classification layer
    model.eval()
    model.to(device)
    return model


def get_activations(images, model, device):
    """
    Calculate activations for a batch of images using Inception v3.
    Expects images in [B, C, H, W] format with values normalized to [0, 1].
    """
    images = images.to(device)

    # Converting grayscale to RGB for MNIST
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    # Resizing to match the v3 input size
    images = nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

    # Normalizing to match the v3 preprocessing
    images = (images - 0.5) / 0.5  # Normalize to [-1, 1]

    # Getting the activations
    activations = model(images)
    return activations.cpu().numpy()


def compute_fid(activations_real_path, generated_images, inception_model, device):
    """
    Calculate FID score between the real and generated images.
    """
    # Loading real image activations
    activations_real = torch.load(activations_real_path)
    mu_real, sigma_real = activations_real.mean(axis=0), np.cov(activations_real, rowvar=False)

    # Preparing fake image activations
    generated_images_tensor = torch.tensor(generated_images).permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    activations_fake = get_activations(generated_images_tensor, inception_model, device)
    mu_fake, sigma_fake = activations_fake.mean(axis=0), np.cov(activations_fake, rowvar=False)

    # Computing the FID score
    diff = mu_real - mu_fake
    covmean, _ = sqrtm(sigma_real @ sigma_fake, disp=False)

    # Handling numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid
