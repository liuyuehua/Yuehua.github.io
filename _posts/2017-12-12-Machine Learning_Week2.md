---
title: Macine Learning_Week2
date: 2017-12-12
categories:
- Machine Learning
tags: 
- Machine Learning
description: Machine Learning Week2 Notes. 
mathjax: true
---
## Linear Regression with Multiple Variables

$$ \begin{align*}x_j^{(i)} &= \text{value of feature } j \text{ in the }i^{th}\text{ training example} \newline x^{(i)}& = \text{the column vector of all the feature inputs of the }i^{th}\text{ training example} \newline m &= \text{the number of training examples} \newline n &= \left| x^{(i)} \right| ; \text{(the number of features)} \end{align*} $$

**Hypothesis function**:

$$ h_\theta (x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n $$

$$\begin{align*}h_\theta(x) =\begin{bmatrix}\theta_0 \hspace{2em}  \theta_1 \hspace{2em}  ...  \hspace{2em}  \theta_n\end{bmatrix}\begin{bmatrix}x_0 \newline x_1 \newline \vdots \newline x_n\end{bmatrix}= \theta^T x\end{align*}$$

The training examples are stored in X row-wise, like such:

$$\begin{align*}X = \begin{bmatrix}x^{(1)}_0 & x^{(1)}_1  \newline x^{(2)}_0 & x^{(2)}_1  \newline x^{(3)}_0 & x^{(3)}_1 \end{bmatrix}&,\theta = \begin{bmatrix}\theta_0 \newline \theta_1 \newline\end{bmatrix}\end{align*}$$

$$h_\theta(X) = X \theta$$

### Cost function

$$ J(\theta) = \dfrac {1}{2m} \displaystyle \sum_{i=1}^m \left (h_\theta (x^{(i)}) - y^{(i)} \right)^2 $$

**The vectorized version is:**

$$ J(\theta) = \dfrac {1}{2m} (X\theta - \vec{y})^{T} (X\theta - \vec{y}) $$

### Gradient Descent for Multiple Variables

$$ \begin{align*}
& \text{repeat until convergence:} \; \lbrace \newline 
\; & \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_0^{(i)}\newline
\; & \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_1^{(i)} \newline
\; & \theta_2 := \theta_2 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_2^{(i)} \newline
& \cdots
\newline \rbrace
\end{align*} $$

In other words:

$$ \begin{align*}& \text{repeat until convergence:} \; \lbrace \newline \; & \theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \;  & \text{for j := 0..n}\newline \rbrace\end{align*} $$

#### Matrix Notation
$$ \theta := \theta - \alpha \nabla J(\theta) $$

$$ \nabla J(\theta)  = \begin{bmatrix}\frac{\partial J(\theta)}{\partial \theta_0}   \newline \frac{\partial J(\theta)}{\partial \theta_1}   \newline \vdots   \newline \frac{\partial J(\theta)}{\partial \theta_n} \end{bmatrix} $$

$$ \begin{align*}
\; &\frac{\partial J(\theta)}{\partial \theta_j} &=&  \frac{1}{m} \sum\limits_{i=1}^{m}  \left(h_\theta(x^{(i)}) - y^{(i)} \right) \cdot x_j^{(i)} \newline
\; & &=& \frac{1}{m} \sum\limits_{i=1}^{m}   x_j^{(i)} \cdot \left(h_\theta(x^{(i)}) - y^{(i)}  \right) 
\end{align*} $$

$$ \begin{align*}\; &\frac{\partial J(\theta)}{\partial \theta_j} &=& \frac1m  \vec{x_j}^{T} (X\theta - \vec{y}) \newline\newline\newline\; &\nabla J(\theta) & = & \frac 1m X^{T} (X\theta - \vec{y}) \newline\end{align*} $$

Finally, the matrix notation (vectorized) of the Gradient Descent rule is:
$$ \theta := \theta - \frac{\alpha}{m} X^{T} (X\theta - \vec{y}) $$