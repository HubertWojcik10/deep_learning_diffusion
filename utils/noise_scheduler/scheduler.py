from typing import Tuple
from torch import Tensor
import torch

class NoiseScheduler:
    """
        Scheduler class that gradually adds noise to images.
    """
    def __init__(self, num_timesteps: float, beta_start: float, beta_end: float) -> Tuple[Tensor, Tensor]:
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps) 
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def add_noise(self, batch: Tensor, t:int):
        """
            Sample x_t from q(x_t | x_0) by taking the original image, a timestep and a precomputed cumulative product. 
            Return a batch with added Gaussian noise.
        """
        # creating Gaussian noise; x_0 size tensor of random numbers from a uniform distribution [0,1)
        noise = torch.randn_like(batch)
        
        # computing the scaling factors
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)   # this information tells us how much of the original image remains after t steps of noising
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1) # .view() changes the shape so that it fits (batch size, channels, height, width)

        # forward process formula
        x_t = sqrt_alpha_bar_t * batch + sqrt_one_minus_alpha_bar_t * noise

        return x_t, noise
