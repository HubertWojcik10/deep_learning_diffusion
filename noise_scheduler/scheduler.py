from typing import Tuple
from torch import Tensor
import torch

class NoiseScheduler:
    """
        Scheduler class that gradually adds noise to images.
    """
    def __init__(self, device, num_timesteps: float, beta_start: float, beta_end: float) -> Tuple[Tensor, Tensor]:
        self.device = device
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps).to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
    
    def add_noise(self, batch: Tensor, t:int):
        """
            Sample x_t from q(x_t | x_0) by taking the original image, a timestep and a precomputed cumulative product. 
            Return a batch with added Gaussian noise.
        """
        # creating Gaussian noise; x_0 size tensor of random numbers from a uniform distribution [0,1)
        noise = torch.randn_like(batch)
        
        # computing the scaling factors
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1).to(self.device)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1).to(self.device)

        # forward process formula
        x_t = sqrt_alpha_bar_t * batch + sqrt_one_minus_alpha_bar_t * noise

        return x_t, noise

    def sample_prev_timestep(self, xt: Tensor, noise_pred: Tensor, t: int):
        """
            Sample the previous timestep x_{t-1} from the current timestep x_t and predicted the noise.
        """
        # reconstructing x_0 from x_t and noise_pred
        x0 = (xt - (self.sqrt_one_minus_alpha_bar[t] * noise_pred)) / self.sqrt_alpha_bar[t]
        x0 = torch.clamp(x0, -1, 1) # normalizing

        # computing the mean of the posterior q(x_{t-1} | x_t, x_0)
        mean = xt - ((self.beta[t] * noise_pred) / self.sqrt_one_minus_alpha_bar[t])
        mean = mean / torch.sqrt(self.alpha[t])

        # handling the x_0 original image case
        if t == 0:
            return mean, x0
        else:
            # coming variance for q(x_{t-1} | x_t, x_0)
            variance = (1 - self.alpha_bar[t-1]) / (1 - self.alpha_bar[t])
            variance = variance * self.beta[t]
            sigma = variance ** 0.5

            # sampling z ~ N(0, I)
            z = torch.randn(xt.shape).to(self.device)

        return mean + sigma * z, x0