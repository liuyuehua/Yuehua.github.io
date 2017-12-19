---
title: Planar data classification with one hidden layer
date: 2017-12-19
categories:
- Deep Learning
tags: 
- Deep Learning
- Neural Networks and Deep Learning
- deeplearning.ai
description: The first neural network, which will have a hidden layer.
mathjax: true
---
## Neural Network model
**Here is our model:**  
<div  align="center"> 
<img src="http://p153fvp85.bkt.clouddn.com/classification_kiank.png" style="width:600px;height:300px;"> 
</div> 

**Mathematically:**  
For one example $x^{(i)}$:
$$z^{[1] (i)} =  W^{[1]} x^{(i)} + b^{[1] (i)}\tag{1}$$ 
$$a^{[1] (i)} = \tanh(z^{[1] (i)})\tag{2}$$
$$z^{[2] (i)} = W^{[2]} a^{[1] (i)} + b^{[2] (i)}\tag{3}$$
$$\hat{y}^{(i)} = a^{[2] (i)} = \sigma(z^{ [2] (i)})\tag{4}$$
$$y^{(i)}_{prediction} = \begin{cases} 1 & \mbox{if } a^{[2](i)} > 0.5 \\ 0 & \mbox{otherwise } \end{cases}\tag{5}$$
  
Given the predictions on all the examples, you can also compute the cost $J$ as follows: 
$$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right)  \large  \right) \small \tag{6}$$  

###  Defining the neural network structure  
Define three variables:
    - n_x: the size of the input layer
    - n_h: the size of the hidden layer (set this to 4) 
    - n_y: the size of the output layer  

### Initialize the model's parameters  
```python
def initialize_parameters(n_x, n_h, n_y):
```

```python
parameters = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2}
    
return parameters
```

### Forward Propagation  
```python
def forward_propagation(X, parameters):
```  

```python
cache = {"Z1": Z1,
         "A1": A1,
         "Z2": Z2,
         "A2": A2}
    
return A2, cache
```  

### Compute Cost  
Now that you have computed $A^{[2]}$ (in the Python variable "`A2`"), which contains $a^{[2](i)}$ for every example, you can compute the cost function as follows:

$$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large{(} \small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right) \large{)} \small\tag{13}$$

Implement `compute_cost()` to compute the value of the cost $J$.

**Instructions**:
- Implement the cross-entropy loss. 
$$ - \sum\limits_{i=0}^{m}  y^{(i)}\log(a^{[2](i)}) $$  

```python
def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    
    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), (1-Y))
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost
```

### Backpropagation  

**Instructions**:
Backpropagation is usually the hardest (most mathematical) part in deep learning. To help you, here again is the slide from the lecture on backpropagation. You'll want to use the six equations on the right of this slide, since you are building a vectorized implementation.  

<img src="http://p153fvp85.bkt.clouddn.com/grad_summary.png" style="width:600px;height:300px;" align=center>  

```python
def backward_propagation(parameters, cache, X, Y):
```  

```python
grads = {"dW1": dW1,
         "db1": db1,
         "dW2": dW2,
         "db2": db2}
    
return grads
```  

**General gradient descent rule**: $ \theta = \theta - \alpha \frac{\partial J }{ \partial \theta }$ where $\alpha$ is the learning rate and $\theta$ represents a parameter.

**Illustration**: The gradient descent algorithm with a good learning rate (converging) and a bad learning rate (diverging). Images courtesy of Adam Harley.  

<img src="http://p153fvp85.bkt.clouddn.com/sgd.gif" style="width:400;height:400;">  <img src="http://p153fvp85.bkt.clouddn.com/sgd_bad.gif" style="width:400;height:400;">  

### Update Parameters  
```python
 W1 = W1 - learning_rate * dW1
 b1 = b1 - learning_rate * db1
 W2 = W2 - learning_rate * dW2
 b2 = b2 - learning_rate * db2
```  

## Build the neural network model in nn_model()  
**Instructions**: The neural network model has to use the previous functions in the right order.  

```python
 for i in range(0, num_iterations):
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)
```  

## Predictions  
**Reminder**: predictions = $$y_{prediction} = \mathbb 1 \text{{activation > 0.5}} = \begin{cases}
      1 & \text{if}\ activation > 0.5 \\
      0 & \text{otherwise}
    \end{cases}$$    

```python
def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    
    return predictions
```  

**Output**: 


<table style="width:40%">
  <tr>
    <td>predictions mean</td>
    <td> 0.666666666667 </td> 
  </tr>
  
</table>  

## Reference
1.[Deep Learning](https://www.deeplearning.ai/)  
2.[Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/)  
3.[Demystifying Deep Convolutional Neural Networks](http://scs.ryerson.ca/~aharley/neural-networks/)  
4.[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-case-study/)
