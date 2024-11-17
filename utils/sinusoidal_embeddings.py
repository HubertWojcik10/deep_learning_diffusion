import torch

def get_sinusoidal_embeddings(timesteps: torch.Tensor, embedding_dim: int = 256, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal time embeddings 
    """
    if embedding_dim % 2 != 0:
        raise ValueError("embedding_dim must be even")
    
    half_dim = embedding_dim // 2
    
    # the exponents array is created by dividing each index by half_dim
    exponents = torch.arange(
        half_dim,
        device=timesteps.device,
        dtype=torch.float32
    ) / half_dim
    
    # each factor is computed as max_period (10000) raised to the power of each exponent
    factors = max_period ** exponents
    
    t_emb = timesteps[:, None].repeat(1, half_dim) / factors

    # any phase-shifted sine wave can be represented as a combination of sine and cosine
    sinusoids = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1) 

    
    return sinusoids