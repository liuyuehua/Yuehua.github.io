---
title: Machine Learning_Week3
date: 2017-12-13
categories:
- Machine Learning
tags: 
- Machine Learning
description: Machine Learning Week3 Notes. 
mathjax: true
---
## Logistic Regression(逻辑回归)
Don't be confused by the name "Logistic Regression", it is named that way for historical reasons and is actually an approach to classification problems, not regression problems. 别被名字误导了，实际是解决分类问题的。
### Binary Classification
**Hypothesis** should satisfy:
$$ 0 \leq h_\theta (x) \leq 1 $$
"**Sigmoid Function**," also called the "**Logistic Function**":
$$ \begin{align*}& h_\theta (x) =  g ( \theta^T x ) \newline \newline& z = \theta^T x \newline& g(z) = \dfrac{1}{1 + e^{-z}}\end{align*} $$

$$h_\theta$$will give us the **probability** that our output is 1 or 0.

二分类满足：
$$ \begin{align*}& h_\theta(x) = P(y=1 | x ; \theta) = 1 - P(y=0 | x ; \theta) \newline& P(y = 0 | x;\theta) + P(y = 1 | x ; \theta) = 1\end{align*} $$

### Decision Boundary
$$\begin{align*}& h_\theta(x) \geq 0.5 \rightarrow y = 1 \newline& h_\theta(x) < 0.5 \rightarrow y = 0 \newline\end{align*}$$

根据**Sigmoid Function** 可以推出：
$$\begin{align*}& \theta^T x \geq 0 \Rightarrow y = 1 \newline& \theta^T x < 0 \Rightarrow y = 0 \newline\end{align*}$$

### Cost Function
Cost function for logistic regression looks like:
$$ \begin{align*}& J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \; & \text{if y = 1} \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x)) \; & \text{if y = 0}\end{align*} $$

The more our hypothesis is off from y, the larger the cost function output. If our hypothesis is equal to y, then our cost is 0:
$$ \begin{align*}& \mathrm{Cost}(h_\theta(x),y) = 0 \text{  if  } h_\theta(x) = y \newline & \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{  if  } y = 0 \; \mathrm{and} \; h_\theta(x) \rightarrow 1 \newline & \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{  if  } y = 1 \; \mathrm{and} \; h_\theta(x) \rightarrow 0 \newline \end{align*} $$

### Simplified Cost Function and Gradient Descent
#### Simplified Cost Function
We can compress our cost function's two conditional cases into one case:
$$ \mathrm{Cost}(h_\theta(x),y) = - y \; \log(h_\theta(x)) - (1 - y) \log(1 - h_\theta(x)) $$
$$ J(\theta) = - \frac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))] $$

A **vectorized** implementation is:
$$ \begin{align*}
& h = g(X\theta)\newline
& J(\theta)  = \frac{1}{m} \cdot \left(-y^{T}\log(h)-(1-y)^{T}\log(1-h)\right)
\end{align*} $$

#### Gradient Descent
$$ \begin{align*}& Repeat \; \lbrace \newline & \; \theta_j := \theta_j - \alpha \dfrac{\partial}{\partial \theta_j}J(\theta) \newline & \rbrace\end{align*} $$

We can work out the derivative part using calculus to get:
$$ \begin{align*}
& Repeat \; \lbrace \newline
& \; \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \newline & \rbrace
\end{align*} $$

A **vectorized** implementation is:
$$ \theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - \vec{y}) $$

#### Partial derivative of J(θ)
The **vectorized** version:
$$ \nabla J(\theta) = \frac{1}{m} \cdot  X^T \cdot \left(g\left(X\cdot\theta\right) - \vec{y}\right) $$

### Advanced Optimization
**Conjugate gradient 
BFGS 
L-BFGS"**

### Multiclass Classification: One-vs-all
$$ \begin{align*}& y \in \lbrace0, 1 ... n\rbrace \newline& h_\theta^{(0)}(x) = P(y = 0 | x ; \theta) \newline& h_\theta^{(1)}(x) = P(y = 1 | x ; \theta) \newline& \cdots \newline& h_\theta^{(n)}(x) = P(y = n | x ; \theta) \newline& \mathrm{prediction} = \max_i( h_\theta ^{(i)}(x) )\newline\end{align*} $$

## Regularization
Regularization is designed to address the problem of **overfitting**.
There are two main options to address the issue of overfitting:
**1. Reduce the number of features:**
- Manually select which features to keep.
- Use a model selection algorithm.

**2. Regularization**
- Keep all the features, but reduce the parameters.
- Regularization works well when we have a lot of slightly useful features.

### Cost Function
$$ min_\theta\ \dfrac{1}{2m}\ \left[ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\ \sum_{j=1}^n \theta_j^2 \right] $$

### Regularized Linear Regression
#### Gradient Descent
$$ \begin{align*}
& \text{Repeat}\ \lbrace \newline
& \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline
& \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\newline
& \rbrace
\end{align*} $$

$$ \theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} $$

#### Normal Equation
$$ \begin{align*}& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \newline& \text{where}\ \ L = \begin{bmatrix} 0 & & & & \newline & 1 & & & \newline & & 1 & & \newline & & & \ddots & \newline & & & & 1 \newline\end{bmatrix}\end{align*} $$

### Regularized Logistic Regression
#### Cost Function
$$ J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))\large] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2 $$

#### Gradient Descent
$$ \begin{align*}& \text{Repeat}\ \lbrace \newline& \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline& \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\newline& \rbrace\end{align*} $$
*$$\theta_0$$不更新*

## Reference
[Machine Learning by Stanford University](https://www.coursera.org/learn/machine-learning/resources/Zi29t)

