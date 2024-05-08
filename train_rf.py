from argparse import ArgumentParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datasets import DATAMODULES
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from differentiable_tree import DifferentiableTree, DifferentiableRandomForest
# import NLLLoss 
from torch.nn import NLLLoss
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

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



if __name__ == "__main__":
    parser = ArgumentParser(prog="VAE for DDF", description="Train differential decision forests")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of estimators in the random forest")
    parser.add_argument("--output-dir", type=str, default=os.path.join("results", 'rf'), help="output directory")
    parser.add_argument("--dataset", type=str, default="mnist", help="dataset to use", choices=DATAMODULES.keys())
    parser.add_argument("--samples-per-image", type=int, default=10, help="number of samples per datapoint")
    parser.add_argument("--embedded-data-path", type=str, default=os.path.join("results", "vae", "mnist_embeddings.npz"), help="path to embedded data to load")
    parser.add_argument("--n-labels", type=int, default=300, help="number of labels in the dataset")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    embedded_data = np.load(args.embedded_data_path)
    mean_train = embedded_data["mean_train"]
    logvar_train = embedded_data["logvar_train"]
    y_train = embedded_data["y_train"]
    mean_val = embedded_data["mean_val"]
    logvar_val = embedded_data["logvar_val"]
    y_val = embedded_data["y_val"]
    mean_test = embedded_data["mean_test"]
    logvar_test = embedded_data["logvar_test"]
    y_test = embedded_data["y_test"]
    
    std_train = np.exp(0.5 * logvar_train)
    X_train = np.concatenate([mean_train, std_train], axis=1)
    X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=args.n_labels, random_state=args.seed, stratify=y_train)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    std_val = np.exp(0.5 * logvar_val)
    X_val = np.concatenate([mean_val, std_val], axis=1)
    #X_val, _, y_val, _ = train_test_split(X_val, y_val, train_size=10, random_state=args.seed, stratify=y_val)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    std_test = np.exp(0.5 * logvar_test)
    X_test = np.concatenate([mean_test, std_test], axis=1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    #variance_scaling_experiments(args, mean_train, logvar_train, y_train, X_test, y_test)
    EPOCHS = 2
    LR = 0.0002
    SCALING_FACTOR = 0.1
    BATCH_SIZE = 16

    X_train_deterministic = X_train[:, :mean_train.shape[1]]
    X_val_deterministic = X_val[:, :mean_train.shape[1]]
    X_test_deterministic = X_test[:, :mean_train.shape[1]]
    #X_val_scaled = X_val.clone()
    #X_val_scaled[:, X_val.shape[1] // 2:] *= VAL_SCALING_FACTOR

    loss_fn = NLLLoss()
    rf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed)
    rf.fit(X_train_deterministic, y_train)
    #print(f"Untuned accuracy: {acc_untuned}")

    drf = DifferentiableRandomForest(rf)

    optimizer = torch.optim.Adam(drf.parameters(), lr=LR)

    X_train_scaled = X_train.clone()
    X_train_scaled[:, X_train.shape[1] // 2:] *= SCALING_FACTOR
    train_dataset = TensorDataset(X_train_scaled, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    inspected_tree = drf.differentiable_trees[3]
    params_before = [[param.detach().clone() for param in level] for level in inspected_tree.get_parameters_by_level()]

    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        epoch_loss = 0
        drf.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = drf(X_batch)
            batch_loss = loss_fn(pred, y_batch)
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
        train_losses.append(epoch_loss)

        epoch_val_loss = 0
        drf.eval()
        with torch.no_grad():
            pred = drf.predict_proba_deterministic(X_val_deterministic)
            epoch_val_loss += loss_fn(pred, y_val)
        val_losses.append(epoch_val_loss)

    params_after = [[param.detach().clone() for param in level] for level in inspected_tree.get_parameters_by_level()]

    for i, (level_before, level_after) in enumerate(zip(params_before, params_after)):
        for j, (param_before, param_after) in enumerate(zip(level_before, level_after)):
            print(f"Level {i}, Node {j}")
            print("Difference in weights: ", (param_before - param_after))
    
    for i, (level_before, level_after) in enumerate(zip(params_before, params_after)):
        differences = []
        for j, (param_before, param_after) in enumerate(zip(level_before, level_after)):
            differences.append(torch.norm(param_before - param_after))
        print(f"Level {i}, Average difference: {sum(differences) / len(differences)}")
    

    drf.eval()
    y_pred_tuned = drf.predict_proba_deterministic(X_test_deterministic)
    acc_tuned = accuracy_score(y_test, y_pred_tuned.argmax(dim=1))
    loss_tuned = loss_fn(y_pred_tuned, y_test)

    rf.fit(X_train_deterministic, y_train)
    y_pred_untuned = rf.predict(X_test_deterministic)
    acc_untuned = accuracy_score(y_test, y_pred_untuned)
    loss_untuned = loss_fn(torch.tensor(y_pred_untuned, dtype=torch.float32), y_test)

    print(f"Accuracy improvement: {acc_tuned - acc_untuned}")
    print(f"Loss improvement: {loss_untuned - loss_tuned}")
