from sklearn.tree import DecisionTreeClassifier
from typing import List, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from torch.distributions.normal import Normal
import torch
from torch import nn
from sklearn.utils.validation import check_is_fitted

class DifferentiableTreeNode:
    def __init__(self, node_id : int | None = None, feature : int | None = None, threshold : torch.Tensor | None = None, value : torch.Tensor | None = None, children : List[Any] | None = None, level : int = 0):
        self.level = level
        self.node_id = node_id
        self.feature = feature
        self.threshold = nn.Parameter(threshold) if threshold is not None else None
        self.value = value
        self.children = children

    def get_depth(self):
        if self.children is None:
            return 0
        return 1 + max(child.get_depth() for child in self.children)
    
    def predict_proba(self, X : torch.Tensor) -> torch.Tensor:
        """
        X : torch.Tensor
            The input data to predict on, shape (n_samples, n_features x 2) where the first half of the features are the mean and the second half are the std.
        """
        if self.feature is None:
            return self.value
        else:
            left_weight = self.left_weight(X[:, self.feature], X[:, self.feature + X.shape[1] // 2], self.threshold).unsqueeze(1)
            left_prediction = self.children[0].predict_proba(X)
            right_prediction = self.children[1].predict_proba(X)
            return left_weight * left_prediction + (1 - left_weight) * right_prediction
    
    def left_weight(self, mean, std, threshold):
        distribution = Normal(mean, std)
        return distribution.cdf(threshold)
    
    def predict_proba_deterministic(self, X : torch.Tensor):
        """
        X : torch.Tensor
            The input data to predict on, shape (n_samples, n_features). Note that the input should not contain the standard deviation.
        """
        if self.feature is None:
            return self.value
        else:
            left_weight = X[:, self.feature] <= self.threshold
            left_weight = left_weight.float().unsqueeze(1)
            left_prediction = self.children[0].predict_proba_deterministic(X)
            right_prediction = self.children[1].predict_proba_deterministic(X)
            return left_weight * left_prediction + (1 - left_weight) * right_prediction
        

    def get_tree_parameters(self):
        if self.feature is not None:
            yield self.threshold
        if(self.children is not None): 
            for child in self.children:
                yield from child.get_tree_parameters()
        



class DifferentiableTree(nn.Module):
    def __init__(self, dt : DecisionTreeClassifier):
        """
        Wraps an sklearn decision tree and provides its tree structure.
        Enables predictions with variations of the step function. 
        """
        super().__init__()
        check_is_fitted(dt)
        self.sklearn_tree = dt
        self.differentiable_tree = self._make_tree(0)
        self.tree_thresholds = nn.ParameterList([threshold for threshold in self.differentiable_tree.get_tree_parameters()])

    def __str__(self):
        return self._str_tree(self.differentiable_tree)
    
    def _str_tree(self, node : DifferentiableTreeNode, depth = 0):
        if node is None:
            return ""
        result = f"{'  ' * depth}Node {node.node_id}"
        if node.feature is not None:
            result += f": feature {node.feature} <= {node.threshold}\n"
            for child in node.children:
                result += self._str_tree(child, depth + 1)
        else:
            result += f": value {node.value}\n"
        return result
    
    def _make_tree(self, node_id : int, level = 0):
        tree = self.sklearn_tree.tree_
        if tree.children_left[node_id] != -1:
            children = [
                self._make_tree(tree.children_left[node_id], level + 1),
                self._make_tree(tree.children_right[node_id], level + 1)
            ]
        else:
            return DifferentiableTreeNode(node_id = node_id, value = torch.tensor(tree.value[node_id][0]), level = level)
        return DifferentiableTreeNode(node_id = node_id, feature = tree.feature[node_id], threshold = torch.tensor(tree.threshold[node_id]), children = children, level = level)
    
    def predict_proba(self, X : np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        X : np.ndarray
            The input data to predict on, shape (n_samples, n_features x 2) where the first half of the features are the mean and the second half are the std.
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if(X.shape[1] != self.sklearn_tree.n_features_in_ * 2):
            raise ValueError(f"Input data has shape {X.shape} but expected shape (n_samples, {self.sklearn_tree.n_features_ * 2}). Note that the first half of the features should be the mean and the second half should be the std.")
        if(torch.any(X[:, X.shape[1] // 2:].min() <= 0)):
            raise ValueError("Standard deviation cannot be non-positive.")
        prediction = self.differentiable_tree.predict_proba(X)
        return prediction / torch.sum(prediction, dim = 1, keepdim=True)

    def predict_proba_deterministic(self, X : np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        X : np.ndarray
            The input data to predict on, shape (n_samples, n_features). Note that the input should not contain the standard deviation.
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if(X.shape[1] != self.sklearn_tree.n_features_in_):
            raise ValueError(f"Input data has shape {X.shape} but expected shape (n_samples, {self.sklearn_tree.n_features_in_}).")
        prediction = self.differentiable_tree.predict_proba_deterministic(X)
        return prediction / torch.sum(prediction, dim = 1, keepdim=True)

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        return self.predict_proba(X)
    
    def get_parameters_by_level(self):
        """
        Returns the thresholds of the tree by level.
        """
        current_nodes = []
        next_nodes = [self.differentiable_tree]
        thresholds_by_level = []
        while next_nodes:
            current_nodes = next_nodes
            next_nodes = []
            current_level_thresholds = [node.threshold for node in current_nodes if node.feature is not None]
            if current_level_thresholds:
                thresholds_by_level.append(current_level_thresholds)
            for node in current_nodes:
                if node.children is not None:
                    next_nodes.extend(node.children)
        return thresholds_by_level

    def get_level_parameters(self, level):
        """
        Returns the thresholds of the tree by level.
        """
        current_nodes = []
        next_nodes = [self.differentiable_tree]
        result = []
        while next_nodes:
            current_nodes = next_nodes
            next_nodes = []
            for node in current_nodes:
                if node.level == level:
                    result.append(node.threshold)
                if node.level < level and node.children is not None:
                    next_nodes.extend(node.children)
        return result
    
class DifferentiableRandomForest(nn.Module):
    def __init__(self, rf : RandomForestClassifier):
        super().__init__()
        self.differentiable_trees = nn.ModuleList([DifferentiableTree(dt) for dt in rf.estimators_])
        self.rf = rf
    
    
    def predict_proba(self, X : np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        X : np.ndarray
            The input data to predict on, shape (n_samples, n_features x 2) where the first half of the features are the mean and the second half are the std.
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        predictions = torch.stack([tree.predict_proba(X) for tree in self.differentiable_trees], dim=2)
        return torch.mean(predictions, dim=2)
    
    def predict_proba_deterministic(self, X : np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        X : np.ndarray
            The input data to predict on, shape (n_samples, n_features). Note that the input should not contain the standard deviation.
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        predictions = torch.stack([tree.predict_proba_deterministic(X) for tree in self.differentiable_trees], dim=2)
        return torch.mean(predictions, dim=2)
    
    def forward(self, X : torch.Tensor) -> torch.Tensor:
        return self.predict_proba(X)

if __name__ == "__main__":
    dataset = make_classification(n_samples=200, n_features=4, random_state=0, shuffle=False)
    X, y = dataset
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)
    dt_wrapper = DifferentiableTree(clf)
    
    X_with_std = np.concatenate([X, np.ones_like(X) * 0.0001], axis=1)

    clf_pred = clf.predict_proba(X)
    ddt_pred = dt_wrapper.predict_proba(torch.tensor(X_with_std, dtype=torch.float32))
    print(np.allclose(clf_pred, ddt_pred.detach().numpy()))
    print(np.all(clf_pred==ddt_pred.detach().numpy()))

    rf = RandomForestClassifier(random_state=0)
    rf.fit(X, y)
    rf_wrapper = DifferentiableRandomForest(rf)
    rf_pred = rf.predict_proba(X)
    rf_wrapper_pred = rf_wrapper.predict_proba(X_with_std).detach().numpy()
    print(np.allclose(rf_pred, rf_wrapper_pred))
    print(np.all(rf_pred==rf_wrapper_pred))

    
    optimizer = torch.optim.Adam(dt_wrapper.differentiable_tree.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    for i in range(100):
        optimizer.zero_grad()
        pred = dt_wrapper.differentiable_tree(torch.tensor(X_with_std, dtype=torch.float32))
        loss = criterion(pred, torch.tensor(y))
        loss.backward()
        optimizer.step()
        print(loss.item())
    
    from sklearn.metrics import accuracy_score
    print("tuned: ", accuracy_score(y, torch.argmax(pred, dim=1).detach().numpy()))
    print("original: ", accuracy_score(y, clf.predict(X)))
    print("tuned: ", accuracy_score(y, rf_wrapper.predict(X)))
    print("original: ", accuracy_score(y, rf.predict(X)))
