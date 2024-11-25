import torch


class NoiseScheduler:
    def __init__(self, t, beta_start, beta_end):
        self.t = t
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.betas = torch.linspace(beta_start, beta_end, t)
        self.alphas = 1 - self.betas
        self.alphas_cum = torch.cumprod(self.alphas, 0)
        self.sqrt_alpha_cum = torch.sqrt(self.alphas_cum)
        self.sqrt_1_minus_alpha_cum = torch.sqrt(1 - self.alphas_cum)
    
    def add_noise(self, batch, noise, t):
        """ Add noise to the batch at timestep t """

        batch_shape = batch.shape
        batch_size = batch_shape[0]

        sqrt_alpha_cum = self.sqrt_alpha_cum[t].reshape(batch_size)
        sqrt_1_minus_alpha_cum = self.sqrt_1_minus_alpha_cum[t].reshape(batch_size)

        for _ in range(len(batch_shape)-1):
            sqrt_alpha_cum = sqrt_alpha_cum.unsqueeze(-1)
            sqrt_1_minus_alpha_cum = sqrt_1_minus_alpha_cum.unsqueeze(-1)
        
        return sqrt_alpha_cum * batch + sqrt_1_minus_alpha_cum * noise
    
    def sample_prev_timestep(self, xt, noise_pred, t):
        """ Sample the previous timestep from the current timestep """
        
        x0 = (xt - (self.sqrt_1_minus_alpha_cum[t] * noise_pred)) / self.sqrt_alpha_cum[t]
        x0 = torch.clamp(x0, -1, 1)

        mean = xt - ((self.betas[t] * noise_pred) /(self.sqrt_1_minus_alpha_cum[t]))
        mean = mean /torch.sqrt(self.alphas[t])

        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alphas_cum[t-1]) / (1 - self.alphas_cum[t])
            variance = variance * self.betas[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
        
        return mean + sigma*z, x0