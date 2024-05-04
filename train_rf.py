from argparse import ArgumentParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datasets import DATAMODULES
import os
import torch
from models import ConvVAEModule
import numpy as np


def embed(vae, dataloader, deterministic=False, n_samples=30):
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
    for i in range(args.samples_per_image):
        std = torch.exp(0.5 * logvar)
        X.append(torch.normal(mean, std))
    X = torch.cat(X, dim=0)
    y = y.repeat(args.samples_per_image)
    return X, y

if __name__ == "__main__":
    parser = ArgumentParser(prog="VAE for DDF", description="Train VAE for downstream use with differential decision forests")
    parser.add_argument("--n-trees", type=int, default=200, help="batch size")
    parser.add_argument("--output-dir", type=str, default=os.path.join("results", 'rf'), help="output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint to load")
    parser.add_argument("--dataset", type=str, default="mnist", help="dataset to use", choices=DATAMODULES.keys())
    parser.add_argument("--samples-per-image", type=int, default=10, help="number of samples per datapoint")
    parser.add_argument("--embedded-data-path", type=str, default=None, help="embedded data to load")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if(args.embedded_data_path is not None):
        embedded_data = np.load(args.embedded_data_path)
        X_train, y_train = embedded_data["X_train"], embedded_data["y_train"]
        X_test, y_test = embedded_data["X_test"], embedded_data["y_test"]
    else:
        vae = ConvVAEModule.load_from_checkpoint(args.checkpoint).vae
        vae.eval()
        datamodule = DATAMODULES[args.dataset](batch_size=512)
        datamodule.setup(None)
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        X_train, y_train = embed(vae, train_dataloader, n_samples=args.samples_per_image)
        X_test, y_test = embed(vae, val_dataloader, deterministic=True)
        np.savez(os.path.join(args.output_dir, "embeddings"), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        X_train_deterministic, y_train_deterministic = embed(vae, train_dataloader, deterministic=True)
    
    rf = RandomForestClassifier(n_estimators=args.n_trees)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Multi-sample Accuracy: {acc}")

    rf_deterministic = RandomForestClassifier(n_estimators=args.n_trees)
    rf_deterministic.fit(X_train_deterministic, y_train_deterministic)
    y_pred = rf_deterministic.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Deterministic Accuracy: {acc}")

