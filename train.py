import yaml

from src.architecture.unet import Unet3D
from src.diffusion.gaussian_diffusion import GaussianDiffusion
from src.trainer.trainer import Trainer

# Load config from YAML
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Model definition
model = Unet3D(**config['model'])
device = "cuda:1"
# Diffusion model
diffusion = GaussianDiffusion(
    denoise_fn = model,
    **config['diffusion']
).to(device)

# Trainer
trainer = Trainer(
    diffusion_model = diffusion,
    **config['trainer'],
    folder = config['training_data_dir']
)

# Start Training
trainer.train()