from argparse import ArgumentParser
import os
from datasets import DATAMODULES
from models import ConvVAEModule
import numpy as np
import torch

def embed_dataloader(vae, dataloader):
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
    return mean, logvar, y

def embed_dataset(vae, dataset, batch_size, output_path):
    vae.eval()
    datamodule = DATAMODULES[dataset](batch_size=batch_size)
    datamodule.setup(None)
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    mean_train, logvar_train, y_train = embed_dataloader(vae, train_dataloader)
    mean_val, logvar_val, y_val = embed_dataloader(vae, val_dataloader)
    mean_test, logvar_test, y_test = embed_dataloader(vae, test_dataloader)
    np.savez(output_path, mean_train=mean_train, logvar_train=logvar_train, y_train=y_train, mean_val=mean_val, logvar_val=logvar_val, y_val=y_val, mean_test=mean_test, logvar_test=logvar_test, y_test=y_test)

if __name__ == "__main__":
    parser = ArgumentParser(prog="VAE embedding function", description="Use trained VAE for embedding data")
    parser.add_argument("--batch-size", type=int, default=512, help="batch size")
    parser.add_argument("--output-dir", type=str, default=os.path.join("results", 'vae'), help="output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint to load")
    parser.add_argument("--dataset", type=str, default="mnist", help="dataset to use", choices=DATAMODULES.keys())
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()


    vae = ConvVAEModule.load_from_checkpoint(args.checkpoint).vae
    output_path = os.path.join(args.output_dir, args.dataset + "_embeddings.npz")
    embed_dataset(vae, dataset=args.dataset, batch_size=args.batch_size, output_path=output_path)