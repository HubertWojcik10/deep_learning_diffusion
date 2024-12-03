import torch
import torchvision
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from model.unet import Unet
from noise_scheduler._archived_scheduler import NoiseScheduler
from utils.load_params import load_params


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    xt = torch.randn((config['num_samples'],
                      config['channels_num'],
                      config['image_size'],
                      config['image_size'])).to(device)
    for i in tqdm(reversed(range(config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        # Save x0
        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists(os.path.join(config['task_name'], 'samples')):
            os.mkdir(os.path.join(config['task_name'], 'samples'))
        img.save(os.path.join(config['task_name'], 'samples', 'x0_{}.png'.format(i)))
        img.close()


def infer():
    config = load_params("config.json")
    
    
    # Load model with checkpoint
    model = Unet().to(device)
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.eval()
    
    # Create the noise scheduler
    scheduler = NoiseScheduler(t=config['num_timesteps'],
                                     beta_start=config['beta_start'],
                                     beta_end=config['beta_end'])
    with torch.no_grad():
        sample(model, scheduler, config)


if __name__ == '__main__':
    infer()
