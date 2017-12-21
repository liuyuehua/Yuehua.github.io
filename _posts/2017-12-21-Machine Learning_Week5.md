---
title: Machine Learning_Week5
date: 2017-12-15
categories:
- Machine Learning
tags: 
- Machine Learning
description: Machine Learning Week5 Notes. Neural Networks: Learning.
mathjax: true
---
## Cost Function  
a) $$L$$= total number of layers in the network  
b) $$s_l$$ = number of units (not counting bias unit) in layer l  
c) $$K$$= number of output units/classes  

We denote $$h_\Theta(x)_k$$ as being a hypothesis that results in the $$k^{th}$$ output.
Our cost function for neural networks is going to be a generalization of the one we used for logistic regression.

Recall that the cost function for regularized logistic regression was:  

$$ J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))\large] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2 $$  

For neural networks, it is going to be slightly more complicated:  

$$ \begin{gather*}\large J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2\end{gather*} $$  

## Backpropagation Algorithm  

Given training set $$ \lbrace (x^{(1)}, y^{(1)}) \cdots (x^{(m)}, y^{(m)})\rbrace $$  

Set $$\Delta^{(l)}_{i,j}$$

For training example t =1 to m:  
- Set $$a^{(1)} := x^{(t)}$$  
- Perform forward propagation to compute $$a^{(l)}$$ for $$l=2,3,…,L$$  
- Using $$y^{(t)}$$ compute $$\delta^{(L)} = a^{(L)} - y^{(t)}$$  
- Compute $$\delta^{(L-1)}, \delta^{(L-2)},\dots,\delta^{(2)}$$ using $$\delta^{(l)} = ((\Theta^{(l)})^T \delta^{(l+1)})\ .*\ a^{(l)}\ .*\ (1 - a^{(l)})$$  
- $$\Delta^{(l)}_{i,j} := \Delta^{(l)}_{i,j} + a_j^{(l)} \delta_i^{(l+1)}$$ or with vectorization, $$\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T$$  
- $$ D^{(l)}_{i,j} := \dfrac{1}{m}\left(\Delta^{(l)}_{i,j} + \lambda\Theta^{(l)}_{i,j}\right) $$ If $$j≠0$$   
- $$D^{(l)}_{i,j} := \dfrac{1}{m}\Delta^{(l)}_{i,j}$$  If $$j=0$$  

## Gradient Checking  

We can approximate the derivative of our cost function with:  

$$ \dfrac{\partial}{\partial\Theta}J(\Theta) \approx \dfrac{J(\Theta + \epsilon) - J(\Theta - \epsilon)}{2\epsilon} $$  

With multiple theta matrices:  
$$ \dfrac{\partial}{\partial\Theta_j}J(\Theta) \approx \dfrac{J(\Theta_1, \dots, \Theta_j + \epsilon, \dots, \Theta_n) - J(\Theta_1, \dots, \Theta_j - \epsilon, \dots, \Theta_n)}{2\epsilon} $$  

## Random Initialization  

$$ \epsilon = \dfrac{\sqrt{6}}{\sqrt{\mathrm{Loutput} + \mathrm{Linput}}} $$  

$$ \Theta^{(l)} =  2 \epsilon \; \mathrm{rand}(\mathrm{Loutput}, \mathrm{Linput} + 1)    - \epsilon $$  

## Reference
[Machine Learning by Stanford University](https://www.coursera.org/learn/machine-learning/resources/EcbzQ)



