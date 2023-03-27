import numpy as np

class SoftmaxRegression:
	"""Numpy implementation of softmax regression."""

	def __init__(self):
		pass

	def linear(self, X, W, b):
	    """Calculates the linear model.
	    
	    Args:
	        X: (d, n) d variables by n samples
	        W: (d, k) weight parameters, d dimensions by k classes
	        b: (k,) bias parameters for k classes
	    
	    Returns:
	        Z: (k, n) applied linear model for k classes by n data samples
	    """
	    Z = np.matmul(W.T, X) + b.reshape(-1, 1)
	    return Z


	def softmax(self, Z):
	    """Defines the softmax function.
	    
	    Args:
	        Z: (k, n) applied linear model for k classes by n samples
	    
	    Returns:
	        softmax: (k, n) estimated probability hypotheses P(y=i|x;w,b)
	            for every i = 1,...,k target for all n samples ie.
	            softmax[k][i] = P(y_i=k|x;w,b), probability that the ith sample belongs to class k
	    """
	    denominator = (np.exp(Z)).sum(axis=0)
	    numerator = np.exp(Z)
	    softmax = numerator / denominator
	    
	    return softmax


	def compute_cost(self, H, Y):
	    """Computes total cost.
	    
	    Args:
	        H: (k, n) estimated softmax probability hypotheses s.t. H[k][i] = P(y_i=k|x;w,b),
	            columns correspond to the likelihood the ith column belongs to the kth class
	        Y: (k, n) one-hot encoded targets of k classes by n samples
	    
	    Returns:
	        cost: (scalar) total cost
	    """
	    loss = (np.multiply(Y, np.log(H))).sum(axis=0)  # (n,) loss for n samples
	    cost = -loss.sum()  # (scalar) total cost
	    
	    return cost


	def gradient_descent(self, X, Y, H, b):
	    """Computes the gradient for softmax regression.

	    Args:
	        X: (d, n) d variables by n samples
	        Y: (k, n) one-hot encoded targets for k classes by n samples
	        H: (k, n) estimated softmax probability hypotheses s.t. H[k][i] = P(y_i=k|x;w,b)
	        b: (k,) bias parameters for k classes

	    Returns:
	        dj_dw: (d, k) gradients of the cost w.r.t. parameters W
	        dj_db: (k,) gradients of cost w.r.t parameter b
	    """
	    n = X.shape[1]
	    k = Y.shape[0]
	    dj_dw = -(1 / n) * np.dot(X, (Y - H).T)
	    dj_db = -(1 / n) * np.sum(Y - H, axis=1)
	    
	    return dj_dw, dj_db.reshape(-1)


	def fit(self, X, Y, alpha, epochs):
	    """Trains the softmax regression model.	    
	    Args: 
	        X: (d, n) d variables by n samples
	        Y: (k, n) one-hot encoded targets for k classes by n samples
	        alpha: (scalar) learning rate for gradient descent
	        epochs: (scalar) number of iterations to train
	        
	    Returns:
	        W: (d, k) fitted parameters for d variables by k classes
	        b: (k,) fitted bias parameter for k classes
	    """
	    d, n = X.shape  # num features x num samples
	    k = Y.shape[0]
	    
	    W_init = np.zeros((d, k))
	    W = W_init
	    b = np.zeros((k,))
	    costs = []
	    
	    for i in range(epochs):
	        Z = self.linear(X, W, b)
	        H = self.softmax(Z)
	        
	        cost = self.compute_cost(H, Y)
	        
	        dj_dw, dj_db = self.gradient_descent(X, Y, H, b)
	        
	        W = W - alpha * dj_dw
	        b = b - alpha * dj_db
	        
	        if i % 100 == 0:
	            costs.append({'epoch': i, 'cost': cost})
	            print(f'epoch: {i}, cost: {cost}')
	    
	    return W, b.reshape(-1)


	def predict(self, X, W, b):
	    """Makes the softmax hypothesis for X given W, b.
	    
	    Args: 
	        X: (d, n) d variables by n samples
	        W: (d, k) parameters for d variables by k classes
	        b: (k,) bias parameter for k classes
	        
	    Returns:
	        H: (k, n) estimated probability hypotheses P(y=i|x;w,b)
	            for every i = 1,...,k target for all n samples ie.
	            softmax[k][i] = P(y_i=k|x;w,b), probability that the ith sample belongs to class k
	    """
	    Z = self.linear(X, W, b)
	    H = self.softmax(Z)
	    
	    return H