import numpy as np

from typing import Optional


class DecisionTreeNode:
    def __init__(self, entropy: float, samples: int):
        self.entropy = entropy
        self.samples = samples

    
class DecisionNode(DecisionTreeNode):
    def __init__(self, feature: int, threshold: int, entropy: float, samples: int, \
                 left: Optional[DecisionTreeNode] = None, right: Optional[DecisionTreeNode] = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        
        super().__init__(entropy, samples)
        
    def __str__(self):
        return f"DecisionNode(feature={self.feature}, threshold={self.threshold}, entropy={self.entropy}, " \
            f"samples={self.samples}, left={self.left}, right={self.right}"

    
class LeafNode(DecisionTreeNode):
    def __init__(self, cls: int, entropy: float, samples: int):
        self.cls = cls
                 
        super().__init__(entropy, samples)
    
    def __str__(self):
        return f"LeafNode(cls={self.cls}, entropy={self.entropy}, samples={self.samples})"


class NotFittedError(Exception):
    ...
    pass


class DecisionTreeClassifier:
    """Numpy implementation of a simple decision tree classifier."""
    
    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2, min_impurity_decrease: float = 0.0):
        """Initializes decision tree classifier instance.
        
        Args:
            max_depth: maximum number of splits in the tree, default None.
            min_samples_split: minimum number of samples required in a node to make a split, default = 2.
            min_impurity_decrease: minimum decrease of impurity to induce a split (greater than or equal), default = 0.0
        """
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_impurity_decrease = min_impurity_decrease
        self._tree = None
        
    
    def fit(self, X, y):
        """Builds the decision tree classifier from the training set (X, y).
        
        Args:
            X: (n, d) n samples by d features.
            y: (n,) targets for n samples.
        """
        self._tree = self._build_tree(X, y, 0)
    
    
    def predict(self, X):
        """Predict classes for X.
        
        Args: 
            X: (n, d) n samples by d features.
        
        Returns:
            y: (n,) predicted classes for n samples.
        
        Raises:
            NotFittedError: If classifier is not yet fitted.
        """
        if self._tree is None: raise NotFittedError("DecisionTreeClassifier instance is not fitted yet. Please "\
                                              "call 'fit' with appropriate arguments first.")
        n_samples = X.shape[0]
        y_hat = []
        
        for i in range(n_samples):
            y_i_hat = self._traverse(X[i], self._tree)
            y_hat.append(y_i_hat)
        
        return np.array(y_hat)
            
        
    def _entropy(self, y):
        """Calculates and returns the entropy of y.
        
        Entropy is calculated as H(X) = sum over all samples from i=1 to n P(X=i)log_2P(X=i).
        If P(X=i) and i = 0 or i = 1 for some class i, entropy = 0.
        
        Args:
            y: (n,) n samples.
        
        Returns:
            entropy: (float) entropy for targets in y.  
        """
        if len(y) == 0: 
            return 0.0

        counts = np.bincount(y)
        p = counts / len(y)
        entropy = -np.sum([p_i * np.log2(p_i) for p_i in p if p_i > 0])
        return entropy
    
    
    def _information_gain(self, y, left_indices, right_indices):
        """Calculates the information gain of a split.
        
        Information gain is calculated as H(node) - weight_left *  H(left) + weight_right * H(right).
        
        Args:
            y: (n,) n samples.
            left_indices: (array-like) indices in left split (subset of y).
            right_indices: (array-like) indices in right split (subset of y).
        
        Returns:
            information_gain: (float) information gain for this split.
            
        """
        if len(left_indices) == 0 or len(right_indices) == 0: 
            return 0
        
        y_left, y_right = y[left_indices], y[right_indices]
        
        entropy_node = self._entropy(y)
        entropy_left = self._entropy(y_left)
        entropy_right = self._entropy(y_right)
        
        w_left, w_right = len(left_indices) / len(y), len(right_indices) / len(y)
        
        information_gain = entropy_node - (w_left * entropy_left + w_right * entropy_right)
        
        return information_gain
        
    
    def _split(self, X_col, threshold):
        """Splits X_col based on threshold.
        
        All samples <= threshold -> left split, all samples > threshold -> right split.
        
        Args:
            X_col: (n,) a single feature column of X.
            threshold: (numeric) a threshold boundary divide samples into two splits.
        
        Returns:
            left_indices: (array-like) indices in X_col belonging to left split.
            right_indices: (array-like) indices in X_col belonging to right split.
        """
        left_indices = np.argwhere(X_col <= threshold).flatten()
        right_indices = np.argwhere(X_col > threshold).flatten()
        
        return left_indices, right_indices
    
    
    def _best_split(self, X, y):
        """Calculates the split with the highest information gain for X.
        
        Args:
            X: (n, d) n samples by d features.
            y: (n,) targets for n samples.
        
        Returns:
            best_feature_index: (int) feature index of feature with highest information gain.
            best_threshold: (numeric) threshold for the feature.
            best_information_gain: (float) information gain for returned feature.
        """
        best_feature_index, best_threshold, best_information_gain = -1, -1, -1
        
        for feature in range(X.shape[1]):
            X_col = X[:, feature]
            
            values = np.unique(X_col)
            for threshold in values:
                left_indices, right_indices = self._split(X_col, threshold)
                information_gain = self._information_gain(y, left_indices, right_indices)
                
                if information_gain > best_information_gain:
                    best_information_gain, best_feature_index, best_threshold = information_gain, feature, threshold
        return best_feature_index, best_threshold, best_information_gain
    
    
    def _build_tree(self, X, y, depth):
        """Builds the decision tree.
        
        Args:
            X: (n, d) n samples by d features where n = # samples at current node.
            y: (n,) targets for n samples.
            depth: (int) depth of current node.
        
        Returns: 
            DecisionTreeNode: a node in the decision tree.
        """
        if depth == self._max_depth or len(y) < self._min_samples_split: 
            return self._create_leaf(y) 
        
        feature, threshold, information_gain = self._best_split(X, y)
        
        if information_gain <= self._min_impurity_decrease: 
            return self._create_leaf(y)
        
        X_col = X[:, feature]
        left_indices, right_indices = self._split(X_col, threshold)
        entropy = self._entropy(y)
        print(f"depth: {depth}, split on feature: {feature}, entropy: {entropy}, left: {left_indices}, right: {right_indices}")
        
        curr = DecisionNode(feature=feature, threshold=threshold, entropy=entropy, samples=len(y))
        
        curr.left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        curr.right = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return curr
    
    
    def _create_leaf(self, y) -> LeafNode:
        """Creates a classification leaf node based on most represented value in y.
        
        Args:
            y: (n,) targets for n samples where n = # samples in the leaf.
        
        Returns: 
            LeafNode: LeafNode with cls as most represented value in y.
        """
        counts = np.bincount(y)
        cls = np.argmax(counts)
        entropy = self._entropy(y)
        return LeafNode(cls, entropy, samples=len(y))
    
    
    def _traverse(self, x, node):
        """Traverses the tree to make a prediction on x.
        
        Args:
            x: (array-like) a single sample in X.
            node: (DecisionTreeNode) current node.
        
        Returns:
            cls: (int) the predicted class for x.
        """
        if type(node) == LeafNode: 
            return node.cls
        
        cls = None
        if x[node.feature] <= node.threshold: cls = self._traverse(x, node.left)
        else: cls = self._traverse(x, node.right)
        
        return cls 