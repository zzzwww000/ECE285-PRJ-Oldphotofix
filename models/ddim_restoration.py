'''
DDIM-based restoration API for project integration.
'''

import torch

def build_timestep_sequence(diffusion_timesteps, teff, device):
    teff = int(teff)
    teff = max(2, min(teff, diffusion_timesteps))
    return torch.linspace(diffusion_timesteps - 1, 0, steps=teff, device=device).long()


def restore_with_ddim(
    model,
    diffusion,
    degraded,
    teff,
    init_mode="img2img",
    strength=0.35,
    x_range=(0.0, 1.0),
):
    """
    DDIM restoration API for project integration.
    Input:
        degraded: tensor [B, 3, H, W]
        teff: int, effective DDIM steps
    Output:
        restored: tensor [B, 3, H, W]
    """
    if degraded.ndim != 4:
        raise ValueError("degraded must be [B, C, H, W]")

    device = degraded.device
    timestep_seq = build_timestep_sequence(diffusion.timesteps, teff, device=device)

    if init_mode == "img2img":
        start_idx = int((len(timestep_seq) - 1) * float(strength))
        start_idx = max(0, min(len(timestep_seq) - 1, start_idx))
        timestep_seq = timestep_seq[start_idx:]
        t_start = timestep_seq[0]
        alpha_hat_start = diffusion.alpha_hat[t_start].view(1, 1, 1, 1)
        x = torch.sqrt(alpha_hat_start) * degraded + torch.sqrt(1 - alpha_hat_start) * torch.randn_like(degraded)
    elif init_mode == "pure_noise":
        x = torch.randn_like(degraded)
    else:
        raise ValueError("init_mode must be 'img2img' or 'pure_noise'")

    x_min, x_max = x_range
    for i in range(len(timestep_seq)):
        t_scalar = int(timestep_seq[i].item())
        t = torch.full((x.shape[0],), t_scalar, device=device, dtype=torch.long)

        with torch.no_grad():
            noise_pred = model(x, degraded, t)

        alpha_hat_t = diffusion.alpha_hat[t].view(-1, 1, 1, 1)

        if i < len(timestep_seq) - 1:
            t_prev_scalar = int(timestep_seq[i + 1].item())
            t_prev = torch.full((x.shape[0],), t_prev_scalar, device=device, dtype=torch.long)
            alpha_hat_prev = diffusion.alpha_hat[t_prev].view(-1, 1, 1, 1)
        else:
            alpha_hat_prev = torch.ones_like(alpha_hat_t)

        x0_pred = (x - torch.sqrt(1 - alpha_hat_t) * noise_pred) / torch.sqrt(alpha_hat_t)
        x0_pred = x0_pred.clamp(x_min, x_max)
        x = torch.sqrt(alpha_hat_prev) * x0_pred + torch.sqrt(1 - alpha_hat_prev) * noise_pred

    return x.clamp(x_min, x_max)
