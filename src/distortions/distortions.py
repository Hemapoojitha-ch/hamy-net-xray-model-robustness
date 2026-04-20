def add_gaussian_noise(x, sigma=0.1):
    return x + sigma * torch.randn_like(x)
