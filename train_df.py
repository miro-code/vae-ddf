from argparse import ArgumentParser
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from differentiable_trees import DifferentiableRandomForest, DifferentiableGradientBoostingClassifier
# import NLLLoss 
from torch.nn import NLLLoss
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd
import json

def sample_from_posterior(mean, logvar, y, deterministic=False, n_samples=5, scaling_factor=1.0):
    if(deterministic):
        return mean, y
    X = []
    for i in range(n_samples):
        std = np.exp(0.5 * logvar)
        std = std * scaling_factor
        X.append(np.random.normal(mean, std))
    X = np.concatenate(X, axis=0)
    y = np.tile(y, reps = n_samples)
    return X, y

def variance_scaling_experiments(args, mean_train, logvar_train, y_train, X_test, y_test):
    X_subsample, _, y_train, _ = train_test_split(np.concatenate([mean_train, logvar_train], axis=1), y_train, train_size=100, random_state=42, stratify=y_train)
    mean_train = X_subsample[:, :mean_train.shape[1]]
    logvar_train = X_subsample[:, mean_train.shape[1]:]
    X_train_deterministic, y_train_deterministic = sample_from_posterior(mean_train, logvar_train, y_train, deterministic=True, n_samples=1)
   
    scaling_factors_std = np.linspace(0.001, 1.3, 150)
    std_results = []
    for scaling_factor in scaling_factors_std:
        X_train, y_train_sampled = sample_from_posterior(mean_train, logvar_train, y_train, deterministic=False, n_samples=args.samples_per_image, scaling_factor=scaling_factor)
        rf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed)
        rf.fit(X_train, y_train_sampled)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Scaling Factor std: {scaling_factor}, Accuracy: {acc}")
        std_results.append(acc)

    scaling_factors_logvar = list(np.linspace(0.001, 3, 20)) + list(np.linspace(3, 6, 130))
    logvar_results = []
    for factor in scaling_factors_logvar:
        X_train, y_train_sampled = sample_from_posterior(mean_train, logvar_train * factor, y_train, deterministic=False, n_samples=args.samples_per_image, scaling_factor=1)
        rf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed)
        rf.fit(X_train, y_train_sampled)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Scaling Factor logvar: {factor}, Accuracy: {acc}")
        logvar_results.append(acc)

    from matplotlib import pyplot as plt
    plt.plot(scaling_factors_std, std_results, color="blue")
    plt.plot(scaling_factors_logvar, logvar_results, color="red")
    plt.legend(["Scaling Factor std", "Scaling Factor logvar"])

    plt.xlabel("Scaling Factor")
    plt.ylabel("Accuracy")
    plt.show()

    rf_deterministic = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed)
    rf_deterministic.fit(X_train_deterministic, y_train_deterministic)
    y_pred = rf_deterministic.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Deterministic Accuracy: {acc}")


def run_experiment(X_train_means, X_std, y_train, X_test_means, y_test, n_estimators, seed, epochs, lr, batch_size, model_name="rf"):

    if model_name == "rf":
        MODEL_CLASS = RandomForestClassifier
        DIFFERENTIABLE_MODEL_CLASS = DifferentiableRandomForest
    elif model_name == "gbt":
        MODEL_CLASS = GradientBoostingClassifier
        DIFFERENTIABLE_MODEL_CLASS = DifferentiableGradientBoostingClassifier

    X_train_full = np.concatenate([X_train_means, X_std], axis=1)

    X_train_means, X_train_full, X_test_means = torch.tensor(X_train_means, dtype=torch.float32), torch.tensor(X_train_full, dtype=torch.float32), torch.tensor(X_test_means, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

    rf = MODEL_CLASS(n_estimators=n_estimators, random_state=seed)
    rf.fit(X_train_means, y_train)
    rf_tuned = DIFFERENTIABLE_MODEL_CLASS(rf)

    optimizer = torch.optim.Adam(rf_tuned.parameters(), lr=lr)
    loss_fn = NLLLoss()
    train_dataset = TensorDataset(X_train_full, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        rf_tuned.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = rf_tuned.predict_proba(X_batch)
            batch_loss = loss_fn(pred, y_batch)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
        train_losses.append(epoch_loss)

        epoch_val_loss = 0
        rf_tuned.eval()
        with torch.no_grad():
            pred = rf_tuned.predict_proba_deterministic(X_test_means)
            epoch_val_loss += loss_fn(pred, y_test)
        val_losses.append(epoch_val_loss)  

    rf_tuned.eval()
    y_pred_tuned = rf_tuned.predict_proba_deterministic(X_test_means)
    acc_tuned = accuracy_score(y_test, y_pred_tuned.argmax(dim=1))
    loss_tuned = loss_fn(y_pred_tuned, y_test).item()

    rf.fit(X_train_means, y_train)
    y_pred_untuned = rf.predict(X_test_means)
    acc_untuned = accuracy_score(y_test, y_pred_untuned)
    loss_untuned = loss_fn(torch.tensor(y_pred_untuned, dtype=torch.float32), y_test).item()

    result = {
        "acc_tuned": acc_tuned,
        "loss_tuned": loss_tuned,
        "acc_untuned": acc_untuned,
        "loss_untuned": loss_untuned,
    }
    return result

def run_all_seeds(embedded_data, epochs, lr, scaling_factor, batch_size, model_name, n_estimators, n_labels):
    
    means_train = embedded_data["mean_train"]
    logvar_train = embedded_data["logvar_train"]
    y_train = embedded_data["y_train"]
    means_test = embedded_data["mean_test"]
    y_test = embedded_data["y_test"]
    

    SEEDS = np.arange(1)

    results = []
    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)
        X_train_split, _, logvar_train_split, _, y_train_split, _ = train_test_split(means_train, logvar_train, y_train, train_size=n_labels, random_state=seed, stratify=y_train)
        X_std_split = np.exp(0.5 * logvar_train_split) * scaling_factor
        run_result = run_experiment(X_train_split, X_std_split, y_train_split, means_test, y_test, n_estimators, seed, epochs, lr, batch_size, model_name)
        results.append([seed, n_estimators, n_labels, epochs, lr, scaling_factor, batch_size, run_result["acc_tuned"], run_result["loss_tuned"], run_result["acc_untuned"], run_result["loss_untuned"]])
    results = pd.DataFrame(results, columns=["seed", "n_estimators", "n_labels", "epochs", "lr", "scaling_factor", "batch_size", "acc_tuned", "loss_tuned", "acc_untuned", "loss_untuned"])
    return results

if __name__ == "__main__":
    parser = ArgumentParser(prog="VAE for DDF", description="Train differential decision forests")
    parser.add_argument("--output-dir", type=str, default=os.path.join("results", 'gbt',"mnist"), help="output directory")
    parser.add_argument("--embedded-data-path", type=str, default=os.path.join("results", "vae", "mnist_embeddings.npz"), help="path to embedded data to load")
    parser.add_argument("--epochs", type=int, default=2, help="num epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--scaling-factor", type=float, default=0.1, help="scaling factor for std")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--n-estimators", type=int, default=100, help="number of estimators")
    parser.add_argument("--n-labels", type=int, default=50, help="number of labels")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    embedded_data = np.load(args.embedded_data_path)
    epochs = args.epochs
    lr = args.lr
    scaling_factor = args.scaling_factor
    batch_size = args.batch_size
    n_estimators = args.n_estimators
    n_labels = args.n_labels

    model_name = "gbt"
    currenct_results = run_all_seeds(embedded_data = embedded_data, epochs = epochs, lr = lr, scaling_factor = scaling_factor, batch_size = batch_size, model_name = model_name, n_estimators = n_estimators, n_labels = n_labels)
    #if resultsfile exists, append to it
    resultsfile = os.path.join(args.output_dir, "results.csv")
    if os.path.exists(resultsfile):
        results = pd.read_csv(resultsfile)
        results = pd.concat([results, currenct_results])
    else:
        results = currenct_results
    results.to_csv(os.path.join(args.output_dir, f"{model_name}_results.csv"), index=False)    

    