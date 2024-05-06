from argparse import ArgumentParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datasets import DATAMODULES
import os
import torch
import numpy as np


def sample_from_posterior(mean, logvar, y, deterministic=False, n_samples=5):
    if(deterministic):
        return mean, y
    X = []
    for i in range(n_samples):
        std = np.exp(0.5 * logvar)
        X.append(np.random.normal(mean, std))
    X = np.concatenate(X, dim=0)
    y = np.repeat(y, repeats = n_samples)
    return X, y

if __name__ == "__main__":
    parser = ArgumentParser(prog="VAE for DDF", description="Train VAE for downstream use with differential decision forests")
    parser.add_argument("--n-trees", type=int, default=200, help="batch size")
    parser.add_argument("--output-dir", type=str, default=os.path.join("results", 'rf'), help="output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint to load")
    parser.add_argument("--dataset", type=str, default="mnist", help="dataset to use", choices=DATAMODULES.keys())
    parser.add_argument("--samples-per-image", type=int, default=5, help="number of samples per datapoint")
    parser.add_argument("--embedded-data", type=str, default=os.path.join("results", "rf", "mnist_embeddings.npz"), help="path to embedded data to load")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    embedded_data = np.load(args.embedded_data)
    mean_train = embedded_data["mean_train"]
    logvar_train = embedded_data["logvar_train"]
    y_train = embedded_data["y_train"]
    X_val = embedded_data["mean_val"]
    y_val = embedded_data["y_val"]
    X_test = embedded_data["mean_test"]
    y_test = embedded_data["y_test"]
    X_train, y_train = sample_from_posterior(mean_train, logvar_train, y_train, deterministic=False, n_samples=args.samples_per_image)
    X_train_deterministic, y_train_deterministic = sample_from_posterior(mean_train, logvar_train, y_train, deterministic=True, n_samples=1)
    
    rf = RandomForestClassifier(n_estimators=args.n_trees, random_state=args.seed)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Multi-sample Accuracy: {acc}")

    rf_deterministic = RandomForestClassifier(n_estimators=args.n_trees, random_state=args.seed)
    rf_deterministic.fit(X_train_deterministic, y_train_deterministic)
    y_pred = rf_deterministic.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Deterministic Accuracy: {acc}")

