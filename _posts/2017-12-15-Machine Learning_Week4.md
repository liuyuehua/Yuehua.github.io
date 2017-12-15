---
title: Machine Learning_Week4
date: 2017-12-15
categories:
- Machine Learning
tags: 
- Machine Learning
description: Machine Learning Week4 Notes. 
mathjax: true
---
## Neural Networks: Representation
### Non-linear Hypotheses
**Neural networks** offers an alternate way to perform machine learning when we have complex hypotheses with many features.
### Neurons and the Brain

### Model Representation I  
$$ \begin{align*}& a_i^{(j)} = \text{"activation" of unit $i$ in layer $j$} \newline& \Theta^{(j)} = \text{matrix of weights controlling function mapping from layer $j$ to layer $j+1$}\end{align*} $$  

If we had one hidden layer, it would look visually something like:  
$$ \begin{bmatrix}x_0 \newline x_1 \newline x_2 \newline x_3\end{bmatrix}\rightarrow\begin{bmatrix}a_1^{(2)} \newline a_2^{(2)} \newline a_3^{(2)} \newline \end{bmatrix}\rightarrow h_\theta(x) $$  

The values for each of the "activation" nodes is obtained as follows:  
$$ \begin{align*}
a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \newline
a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \newline
a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \newline
h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \newline
\end{align*} $$  

Each layer gets its own matrix of weights, $$\Theta^{(j)}$$  

The dimensions of these matrices of weights is determined as follows:
$$ \text{If network has $s_j$ units in layer $j$ and $s_{j+1}$ units in layer $j+1$, then $\Theta^{(j)}$ will be of dimension $s_{j+1} \times (s_j + 1)$.} $$  

### Model Representation II  
####  vectorized implementation  
$$ \begin{align*}a_1^{(2)} = g(z_1^{(2)}) \newline a_2^{(2)} = g(z_2^{(2)}) \newline a_3^{(2)} = g(z_3^{(2)}) \newline \end{align*} $$  

In other words, for layer j=2 and node k, the variable z will be:  
$$ z_k^{(2)} = \Theta_{k,0}^{(1)}x_0 + \Theta_{k,1}^{(1)}x_1 + \cdots + \Theta_{k,n}^{(1)}x_n $$  

The vector representation of x and $$z^{j}$$ is:
$$ \begin{align*}x = \begin{bmatrix}x_0 \newline x_1 \newline\cdots \newline x_n\end{bmatrix} &z^{(j)} = \begin{bmatrix}z_1^{(j)} \newline z_2^{(j)} \newline\cdots \newline z_n^{(j)}\end{bmatrix}\end{align*} $$  

Setting $$x = a^{(1)}$$  we can rewrite the equation as:  
$$ z^{(j)} = \Theta^{(j-1)}a^{(j-1)} $$  

matlab code:  
```
a1 = [ones(m, 1) X]; % add 1
z2 = a1 * Theta1';
a2 = [ones(size(sigmoid(z2), 1), 1) sigmoid(z2)]; % add 1
z3 = a2 * Theta2';
a3 = sigmoid(z3); % H_theta(x)
```

We then get our final result with:
$$ a^{(j)} = g(z^{(j)}) $$  

### Multiclass Classification
To classify data into multiple classes, we let our hypothesis function return a vector of values. Say we wanted to classify our data into one of four final resulting classes:  
$$ \begin{align*}\begin{bmatrix}x_0 \newline x_1 \newline x_2 \newline\cdots \newline x_n\end{bmatrix} \rightarrow\begin{bmatrix}a_0^{(2)} \newline a_1^{(2)} \newline a_2^{(2)} \newline\cdots\end{bmatrix} \rightarrow\begin{bmatrix}a_0^{(3)} \newline a_1^{(3)} \newline a_2^{(3)} \newline\cdots\end{bmatrix} \rightarrow \cdots \rightarrow\begin{bmatrix}h_\Theta(x)_1 \newline h_\Theta(x)_2 \newline h_\Theta(x)_3 \newline h_\Theta(x)_4 \newline\end{bmatrix} \rightarrow\end{align*} $$  

Our resulting hypothesis for one set of inputs may look like:  
$$ h_\Theta(x) =\begin{bmatrix}0 \newline 0 \newline 1 \newline 0 \newline\end{bmatrix} $$  

## Reference
[Machine Learning by Stanford University](https://www.coursera.org/learn/machine-learning/resources/RmTEz)