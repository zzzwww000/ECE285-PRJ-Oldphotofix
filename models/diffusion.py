'''
Diffusion process implementation for image restoration.
- timesteps: int, total diffusion steps T
'''
import torch

class Diffusion:
    """
    Diffusion process implementation for image restoration.
    - timesteps: int, total diffusion steps T
    """

    def __init__(self, timesteps=200, device="cuda"):

        self.timesteps = timesteps
        self.device = device

        # beta schedule
        self.beta = torch.linspace(
            1e-4,
            0.02,
            timesteps
        ).to(device)

        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample_timesteps(self, batch_size):
        """
        Sample random timesteps for a batch of images.
        Output: tensor of shape [batch_size], values in [1, timesteps-1
        """
        return torch.randint(
            low=1,
            high=self.timesteps,
            size=(batch_size,),
            device=self.device
        )

    def noise_images(self, x, t):
        """
        Add noise to clean images x at given timesteps t.
        x = clean image
        t = timestep
        """

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        noise = torch.randn_like(x)

        xt = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise

        return xt, noise