import torch

def embed(vae, dataloader, deterministic=False, n_samples=5):
    mean, logvar, y = [], [], []
    for images, labels in dataloader:
        with torch.no_grad():
            mu, sigma = vae.encode(images)
            mean.append(mu)
            logvar.append(sigma)
            y.append(labels)
    mean = torch.cat(mean, dim=0)
    logvar = torch.cat(logvar, dim=0)
    y = torch.cat(y, dim=0)
    if(deterministic):
        return mean, y
    X = []
    for i in range(n_samples):
        std = torch.exp(0.5 * logvar)
        X.append(torch.normal(mean, std))
    X = torch.cat(X, dim=0)
    y = y.repeat(n_samples)
    return X, y