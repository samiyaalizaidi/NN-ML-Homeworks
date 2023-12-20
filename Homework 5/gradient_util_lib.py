""" 
gradient_util_lib.py
    functions used in gradient descent implementation for linear
    and logistic regression
"""
import numpy as np
import math, copy

#
# Function to perform Zscore normalization
#
def zscore_normalize(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
      
    """
    # find the mean of each column/feature
    mu = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)

#
# Compute the squared-error cost function
#
def compute_cost_matrix(X, y, A): 
    """
    compute cost
    Args:
      X (ndarray (m,n+1)): Feature Matrix, m examples with n features
      y (ndarray (m,)) : target values
      A (ndarray (n+1,)) : model parameters  
      
    Returns:
      cost (scalar): cost
      
    """
    m = X.shape[0]
    
    z = np.dot(X, A.T) - y              
    cost = np.dot(z.T, z)
    
    cost /= (2 * m)                      #scalar    
    return cost

#
# Sigmoind Function
#
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """          
    g = 1.0 /(1 + np.exp(-z))
    
    return g

#
# Prediction for logistic regression
#
def predict_logistic(X, A):
    '''
    Returns 1D array of probabilities that the class label == 1 
  
    Args:
    x (ndarray): Shape (n+1,) input features
    A (ndarray): Shape (n+1,) model parameters   
  
    Returns:
    z (scalar):  prediction
    '''

    y = np.dot(X, A)
    z = sigmoid(y)
    
    return z

#
# Compute the cost function for logistic regression (matrix format) 
#
def compute_cost_logistic(X, y, A):
    """
    compute cost for logistic regression 
    
    Args:
      X (ndarray (m,n+1)): feature matrix, m examples with n+1 features
      y (ndarray (m,)) : target values (0 or 1)
      A (ndarray (n+1,)) : model parameters  
      
    Returns:
      cost (scalar): cost
      
    """
    
    m = X.shape[0]

    pred = predict_logistic(X, A)

    # Calculate the loss when y=1
    
    loss_1 = -np.dot(y, np.log(pred))

    # Calculate the loss when y=0
    loss_2 = -np.dot((1 - y), np.log(1-pred))
    
    #Take the sum of both costs and average 
    cost = (loss_1 + loss_2)/m

    return cost

# 
# Function to calcualte gradient for logistic regression 
#
def compute_gradient_logistic(X, y, A): 
    """
    Computes the gradient for logistic regression 
    Args:
      X (ndarray (m,n+1)): Data, m examples with n features
      y (ndarray (m,)) : target values
      A (ndarray (n+1,)) : model parameters  
      
    Returns:
      dj_da (ndarray (n+1,)): The gradient of the cost w.r.t. the parameters A. 
       
    """
    m, n = X.shape           #(number of examples, number of features)
    
    pred = predict_logistic(X, A)
    
    z = pred - y
    
    dj_da = np.dot(X.T, z)
    
    dj_da /= m                       
        
    return dj_da

#
# Function to calculate the gradient vectors
#
def compute_gradient_vector(X, y, A): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n+1)): Data, m examples with n features
      y (ndarray (m,)) : target values
      A (ndarray (n+1,)) : model parameters  
      
    Returns:
      dj_da (ndarray (n+1,)): The gradient of the cost w.r.t. the parameters A. 
       
    """
    m, n = X.shape           #(number of examples, number of features)
    
    z = np.dot(X, A.T) - y
    dj_da = np.dot(X.T, z)
    
    dj_da /= m                       
        
    return dj_da

#
# Execute the Gradient Descent Update 
#
def gradient_descent(X, y, A_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n+1)) : Feature Matrix, m examples with n+1 features
      y (ndarray (m,))    : target values
      A_in (ndarray (n+1,)) : initial model parameters  
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      A (ndarray (n+1,)) : Updated values of parameters 
      
      """
    
    # An array to store cost J and A at each iteration primarily for graphing later
    J_history = []
    A = copy.deepcopy(A_in)  #avoid modifying global w within function
   
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_da = gradient_function(X, y, A)  

        # Update Parameters using A, alpha and gradient
        A = A - alpha * dj_da        # vector             
        
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append(cost_function(X, y, A))

        # Print cost every at intervals 10 times or as many iterations if < 10
        #if i% math.ceil(num_iters / 10) == 0:
        #    print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"{i:9d} {J_history[-1]:0.5e}")
        
    return A, J_history #return final A, a0 and J history for graphing
