---
title: Macine Learning_Week1
date: 2017-12-12
categories:
- Machine Learning
tags: 
- Machine Learning
description: Machine Learning Week1 Notes.
mathjax: true
---
## Introduction
### What is Machine Learning?
Two definitions of Machine Learning are offered. Arthur Samuel described it as: *"the field of study that gives computers the ability to learn without being explicitly programmed."* This is an older, informal definition.
Tom Mitchell provides a more modern definition: *"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E"*.

In general, any machine learning problem can be assigned to one of two broad classifications:
**Supervised learning(监督学习)** and **Unsupervised learning(非监督学习)**.

### Supervised Learning
1. There is a relationship between the input and the output.
2. Supervised learning problems are categorized into **"regression（回归）"** and **"classification（分类）"** problems.

*In a **regression** problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a **classification** problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.*

### Unsupervised Learning
1. Unsupervised learning allows us to approach problems with little or no idea what our results should look like.
1. We can derive this structure by **clustering** the data based on relationships among the variables in the data.
1. With unsupervised learning there is no feedback based on the prediction results.

## Linear Regression with One Variable(单变量线性回归)

### Cost Function(代价函数)
We can measure the accuracy of our hypothesis function by using a **cost function**.

$$J(\theta_0, \theta_1) = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left ( \hat{y}_{i}- y_{i} \right)^2 = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left (h_\theta (x_{i}) - y_{i} \right)^2 $$

matlab code:
```
h = X*theta;
squareErrors = (h-y) .^2;
J = (1/(2*m))*sum(squareErrors);
```

This function is otherwise called the **"Squared error function"**, or **"Mean squared error"**.

**Goal**:  minimize $$J(\theta_0, \theta_1)$$

### Gradient Descent(梯度下降法)
The gradient descent algorithm is:
**repeat until convergence:**
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)$$
**where $$j=0,1$$ represents the feature index number.**
*At each iteration $$j$$, one should **simultaneously update** the parameters $$\theta_0, \theta_1, ... \theta_n$$. Updating a specific parameter prior to calculating another one on the j(th) iteration would yield to a wrong implementation.*

#### Gradient Descent For Linear Regression:

$$ \begin{align*} \text{repeat until convergence: } \lbrace & \newline \theta_0 := & \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}(h_\theta(x_{i}) - y_{i}) \newline \theta_1 := & \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}\left((h_\theta(x_{i}) - y_{i}) x_{i}\right) \newline \rbrace& \end{align*} $$

So, this is simply gradient descent on the original cost function J. This method looks at every example in the entire training set on every step, and is called **batch gradient descent**.

## Reference
[Machine Learning by Stanford University](https://www.coursera.org/learn/machine-learning/resources/JXWWS)















