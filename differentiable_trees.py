from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.utils.validation import check_is_fitted
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.functional import softmax
from typing import List, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class TreeNode:
    def __init__(self, node_id : int | None = None, feature : int | None = None, threshold : torch.Tensor | None = None, value : torch.Tensor | None = None, children : List[Any] = [], level : int = 0):
        self.level = level
        self.node_id = node_id
        self.feature = feature
        self.threshold = nn.Parameter(threshold) if threshold is not None else None
        self.value = value
        self.children = children

    def get_depth(self):
        if not self.children:
            return 0
        return 1 + max(child.get_depth() for child in self.children)
    
    def left_weight(self, mean, std, threshold):
        distribution = Normal(mean, std)
        return distribution.cdf(threshold)
    
    def get_tree_parameters(self):
        if self.feature is not None:
            yield self.threshold
        if(self.children): 
            for child in self.children:
                yield from child.get_tree_parameters()

    def get_tree_nodes(self):
        if not self.children:
            return [self]
        result = [self]
        for child in self.children:
            result.extend(child.get_tree())
        return result

class RegressorNode(TreeNode):
    def __init__(self, node_id : int | None = None, feature : int | None = None, threshold : torch.Tensor | None = None, value : torch.Tensor | None = None, children : List[Any] | None = None, level : int = 0):
        super().__init__(node_id, feature, threshold, value, children, level)

    def predict(self, X : torch.Tensor) -> torch.Tensor:
        """
        X : torch.Tensor
            The input data to predict on, shape (n_samples, n_features x 2) where the first half of the features are the mean and the second half are the std.
        """
        if self.feature is None:
            return torch.tile(self.value, (X.shape[0], 1))
        else:
            left_weight = self.left_weight(X[:, self.feature], X[:, self.feature + X.shape[1] // 2], self.threshold).unsqueeze(1)
            left_prediction = self.children[0].predict(X)
            right_prediction = self.children[1].predict(X)
            return left_weight * left_prediction + (1 - left_weight) * right_prediction
        
    def predict_deterministic(self, X : torch.Tensor):
        """
        X : torch.Tensor
            The input data to predict on, shape (n_samples, n_features). Note that the input should not contain the standard deviation.
        """
        if self.feature is None:
            return torch.tile(self.value, (X.shape[0], 1))
        else:
            left_weight = X[:, self.feature] <= self.threshold
            left_weight = left_weight.float().unsqueeze(1)
            left_prediction = self.children[0].predict_deterministic(X)
            right_prediction = self.children[1].predict_deterministic(X)
            return left_weight * left_prediction + (1 - left_weight) * right_prediction


class ClassifierNode(TreeNode):

    def __init__(self, node_id : int | None = None, feature : int | None = None, threshold : torch.Tensor | None = None, value : torch.Tensor | None = None, children : List[Any] | None = None, level : int = 0):
        super().__init__(node_id, feature, threshold, value, children, level)

    def predict_proba(self, X : torch.Tensor) -> torch.Tensor:
        """
        X : torch.Tensor
            The input data to predict on, shape (n_samples, n_features x 2) where the first half of the features are the mean and the second half are the std.
        """
        if self.feature is None:
            return torch.tile(self.value, (X.shape[0], 1))
        else:
            left_weight = self.left_weight(X[:, self.feature], X[:, self.feature + X.shape[1] // 2], self.threshold).unsqueeze(1)
            left_prediction = self.children[0].predict_proba(X)
            right_prediction = self.children[1].predict_proba(X)
            return left_weight * left_prediction + (1 - left_weight) * right_prediction
    
    def predict_proba_deterministic(self, X : torch.Tensor):
        """
        X : torch.Tensor
            The input data to predict on, shape (n_samples, n_features). Note that the input should not contain the standard deviation.
        """
        if self.feature is None:
            return torch.tile(self.value, (X.shape[0], 1))
        else:
            left_weight = X[:, self.feature] <= self.threshold
            left_weight = left_weight.float().unsqueeze(1)
            left_prediction = self.children[0].predict_proba_deterministic(X)
            right_prediction = self.children[1].predict_proba_deterministic(X)
            return left_weight * left_prediction + (1 - left_weight) * right_prediction
        

class SKLearnTreeWrapper(nn.Module):
    def __init__(self, dt : DecisionTreeRegressor | DecisionTreeClassifier):
        """
        Wraps an sklearn decision tree and provides its tree structure.
        """
        super().__init__()
        check_is_fitted(dt)
        self.sklearn_tree = dt
        self.NODE_CLASS = RegressorNode if isinstance(dt, DecisionTreeRegressor) else ClassifierNode
        self.differentiable_tree = self._make_tree(0)
        self.tree_thresholds = nn.ParameterList([threshold for threshold in self.differentiable_tree.get_tree_parameters()])
    
    def _make_tree(self, node_id : int, level = 0):
        tree = self.sklearn_tree.tree_
        if tree.children_left[node_id] != -1:
            children = [
                self._make_tree(tree.children_left[node_id], level + 1),
                self._make_tree(tree.children_right[node_id], level + 1)
            ]
        else:
            return self.NODE_CLASS(node_id = node_id, value = torch.tensor(tree.value[node_id][0]), level = level)
        return self.NODE_CLASS(node_id = node_id, feature = tree.feature[node_id], threshold = torch.tensor(tree.threshold[node_id]), children = children, level = level)
    
    def __str__(self):
        return self._str_tree(self.differentiable_tree)
    
    def _str_tree(self, node : RegressorNode, depth = 0):
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
                if node.children:
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
                if node.level < level and node.children:
                    next_nodes.extend(node.children)
        return result
    
class DifferentiableTreeRegressor(SKLearnTreeWrapper):
    def __init__(self, dt : DecisionTreeRegressor):
        """
        Wraps an sklearn decision tree and provides its tree structure.
        """
        super().__init__(dt)


    def predict(self, X : torch.Tensor) -> torch.Tensor:
        return self.differentiable_tree.predict(X)
    
    def predict_deterministic(self, X : torch.Tensor) -> torch.Tensor:
        return self.differentiable_tree.predict_deterministic(X)
    
    
class DifferentiableTreeClassifier(SKLearnTreeWrapper):
    def __init__(self, dt : DecisionTreeClassifier):
        """
        Wraps an sklearn decision tree and provides its tree structure.
        """
        super().__init__(dt)

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
    
    
class DifferentiableRandomForest(nn.Module):
    def __init__(self, rf : RandomForestClassifier):
        super().__init__()
        self.differentiable_trees = nn.ModuleList([DifferentiableTreeClassifier(dt) for dt in rf.estimators_])
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


        
class DifferentiableGradientBoostingRegressor(nn.Module):
    
    def __init__(self, gbr : GradientBoostingRegressor):
        super().__init__()
        self.differentiable_trees = nn.ModuleList([DifferentiableTreeRegressor(dt[0]) for dt in gbr.estimators_])
        self.gbr = gbr
        self.gbr_learning_rate = gbr.learning_rate
    
    def predict(self, X : np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        X : np.ndarray
            The input data to predict on, shape (n_samples, n_features x 2) where the first half of the features are the mean and the second half are the std.
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        predictions = torch.stack([tree.predict(X) for tree in self.differentiable_trees], dim=2)
        return torch.sum(predictions, dim=2) * self.gbr_learning_rate #TODO CHANGE DIM TO 1 IF NECESSARY
    
    def predict_deterministic(self, X : np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        X : np.ndarray
            The input data to predict on, shape (n_samples, n_features). Note that the input should not contain the standard deviation.
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        predictions = torch.stack([tree.predict_deterministic(X) for tree in self.differentiable_trees], dim=2)
        return torch.sum(predictions, dim=2) * self.gbr_learning_rate  #TODO CHANGE DIM TO 1 IF NECESSARY
    
    def forward(self, X : torch.Tensor) -> torch.Tensor:
        return self.predict(X)
    
class DifferentiableGradientBoostingClassifier(nn.Module):
    def __init__(self, gbr : GradientBoostingClassifier):
        super().__init__()
        self.differentiable_trees = nn.ModuleList([nn.ModuleList([DifferentiableTreeRegressor(dt) for dt in class_estimators]) for class_estimators in gbr.estimators_.transpose()])
        self.gbr = gbr
        self.gbr_learning_rate = gbr.learning_rate
    
    def predict_proba(self, X : np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        X : np.ndarray
            The input data to predict on, shape (n_samples, n_features x 2) where the first half of the features are the mean and the second half are the std.
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        class_predictions = []
        for class_id in range(len(self.differentiable_trees)):
            tree_predictions = torch.stack([tree.predict(X) for tree in self.differentiable_trees[class_id]], dim=2)
            class_prediction = torch.sum(tree_predictions, dim=2) * self.gbr_learning_rate #TODO CHANGE DIM TO 1 IF NECESSARY
            class_predictions.append(class_prediction)
        
        result = softmax(torch.stack(class_predictions, dim=1), dim=1)
        return result.squeeze(2)
         
    
    def predict(self, X : np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        X : np.ndarray
            The input data to predict on, shape (n_samples, n_features x 2) where the first half of the features are the mean and the second half are the std.
        """
        return torch.argmax(self.predict_proba(X), dim=1)
    
    def predict_proba_deterministic(self, X : np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        X : np.ndarray
            The input data to predict on, shape (n_samples, n_features x 2) where the first half of the features are the mean and the second half are the std.
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        class_predictions = []
        for class_id in range(len(self.differentiable_trees)):
            tree_predictions = torch.stack([tree.predict_deterministic(X) for tree in self.differentiable_trees[class_id]], dim=2)
            class_prediction = torch.sum(tree_predictions, dim=2) * self.gbr_learning_rate #TODO CHANGE DIM TO 1 IF NECESSARY
            class_predictions.append(class_prediction)
        
        result = softmax(torch.stack(class_predictions, dim=1), dim=1)
        return result.squeeze(2)
    
    def predict_deterministic(self, X : np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        X : np.ndarray
            The input data to predict on, shape (n_samples, n_features x 2) where the first half of the features are the mean and the second half are the std.
        """
        return torch.argmax(self.predict_proba_deterministic(X), dim=1)
    
    


if __name__=="__main__":
    reg_data = make_classification(n_samples=200, n_features=20, n_informative=10, n_redundant=5, n_clusters_per_class=2, random_state=42, n_classes=3)
    X, y = reg_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    gbr = GradientBoostingClassifier(n_estimators=10, random_state=42, init="zero")
    gbr.fit(X_train, y_train)
    pred = gbr.predict_proba(X_test)
    gbr_test = DifferentiableGradientBoostingClassifier(gbr)
    X_std = np.ones_like(X_test) * 0.00001
    X_test_full = torch.tensor(np.concatenate([X_test, X_std], axis=1), dtype=torch.float32)
    pred_test = gbr_test.predict_proba(X_test_full).detach().numpy()
    pred_test_deterministic = gbr_test.predict_proba_deterministic(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
    print(pred - pred_test)
    print(pred == pred_test)
    print(np.isclose(pred_test, pred, atol=1e-10))

    #test rf
    reg_data = make_classification(n_samples=200, n_features=20, n_informative=10, n_redundant=5, n_clusters_per_class=2, random_state=42, n_classes=3)
    X, y = reg_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    pred = rf.predict_proba(X_test)
    rf_test = DifferentiableRandomForest(rf)
    X_std = np.ones_like(X_test) * 0.00001
    X_test_full = torch.tensor(np.concatenate([X_test, X_std], axis=1), dtype=torch.float32)
    pred_test = rf_test.predict_proba(X_test_full).detach().numpy()
    print(pred - pred_test)
    print(pred == pred_test)
    print(np.isclose(pred_test, pred, atol=1e-10))
    

