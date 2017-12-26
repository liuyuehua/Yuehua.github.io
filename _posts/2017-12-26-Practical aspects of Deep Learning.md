---
title: Practical aspects of Deep Learning
date: 2017-12-26
categories:
- Deep Learning
tags: 
- Deep Learning
- Improving Deep Neural Networks 
- deeplearning.ai
description: How to to improve Deep Neural Networks.
mathjax: true
---
## Initialization

A well chosen initialization can:
1. Speed up the convergence of gradient descent  
1. Increase the odds of gradient descent converging to a lower training (and generalization) error   

### He initialization  

随机初始化参数后乘以：

$$ \sqrt{\frac{2}{\text{dimension of the previous layer}}} $$

```python
for l in range(1, L + 1):

        parameters['W' + str(l)] = np.multiply(np.random.randn(layers_dims[l], layers_dims[l-1]), np.sqrt(2./layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
return parameters
```

**Results：**  

<div  align="center">
<img src="http://p153fvp85.bkt.clouddn.com/He.png" style="width:600px;height:400px;">
</div>  

### Conclusions  
- Different initializations lead to different results
- Random initialization is used to break symmetry and make sure different hidden units can learn different things
- Don't intialize to values that are too large
- He initialization works well for networks with ReLU activations.


## Regularization  

### L2 Regularization  

**L2-regularization** relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.

#### The implications of L2-regularization on:

1. **The cost computation:**
A regularization term is added to the cost
1. **The backpropagation function:**
There are extra terms in the gradients with respect to weight matrices
1. **Weights end up smaller ("weight decay"):** Weights are pushed to smaller values.


$$ J_{regularized} = \small \underbrace{-\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} }_\text{cross-entropy cost} + \underbrace{\frac{1}{m} \frac{\lambda}{2} \sum\limits_l\sum\limits_k\sum\limits_j W_{k,j}^{[l]2} }_\text{L2 regularization cost} \tag{2} $$   


**compute_cost_with_regularization:**  
```python
cost = cross_entropy_cost + L2_regularization_cost
```

**backward_propagation_with_regularization:**  
Add the regularization term's gradient  $$\frac{d}{dW} ( \frac{1}{2}\frac{\lambda}{m}  W^2) = \frac{\lambda}{m} W$$  

### Dropout  
Dropout is a widely used regularization technique that is specific to deep learning. **It randomly shuts down some neurons in each iteration.**  
[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf "Dropout: A Simple Way to Prevent Neural Networks from Overfitting")

#### Forward propagation with dropout  
**Steps:**
1. Initialize matrix D1 = np.random.rand(..., ...)  
1. Convert entries of D1 to 0 or 1 (using keep_prob as the threshold)  
1. Shut down some neurons of A1  
1. Scale the value of neurons that haven't been shut down  

```python
 D1 = np.random.rand(A1.shape[0], A1.shape[1])      #Step1                                   
 D1 = D1 < keep_prob                                #Step2
 A1 = A1 * D1                                       #Step3  
 A1 = A1 / keep_prob                                #Step4
```

#### Backward propagation with dropout  
**Steps:**
1. Apply mask D2 to shut down the same neurons as during the forward propagation.  
1. Scale the value of neurons that haven't been shut down.    

```python
dA2 = dA2 * D2              # Step 1
dA2 = dA2 / keep_prob       # Step 2
```  

#### About Dropout:

1. Dropout is a regularization technique.
1. **You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time.**
1. Apply dropout both during forward and backward propagation.
1. **During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations.** For example, if keep_prob is 0.5, then we will on average shut down half the nodes, so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. Dividing by 0.5 is equivalent to multiplying by 2. Hence, the output now has the same expected value. You can check that this works even when keep_prob is other values than 0.5.  

### Conclusions  
1. Regularization will help you reduce overfitting.
1. Regularization will drive your weights to lower values.
1. L2 regularization and Dropout are two very effective regularization techniques.  

## Gradient Checking  

### N-dimensional gradient checking

$$ \frac{\partial J}{\partial \theta} = \lim_{\varepsilon \to 0} \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon} \tag{1} $$

**LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID:**  
<div  align="center">
<img src="http://p153fvp85.bkt.clouddn.com/NDgrad_kiank.png" style="width:800px;height:400px;">
</div>    


**dictionary_to_vector() and vector_to_dictionary():**  
<div  align="center">
<img src="http://p153fvp85.bkt.clouddn.com/dictionary_to_vector.png" style="width:800px;height:400px;">
</div>  

### About Gradient Checking  
1. Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient (computed using forward propagation).
1. Gradient checking is slow, so we don't run it in every iteration of training. You would usually run it only to make sure your code is correct, then turn it off and use backprop for the actual learning process.  

## Reference
1.[Deep Learning](https://www.deeplearning.ai/)  
2.[Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/) 