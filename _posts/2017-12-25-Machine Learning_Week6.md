---
title: Machine Learning_Week6
date: 2017-12-25
categories:
- Machine Learning
tags: 
- Machine Learning
description: Machine Learning Week6 Notes. Advice for Applying Machine Learning&Machine Learning System Design.
mathjax: true
---
## Advice for Applying Machine Learning  

------------
### Evaluating a Hypothesis  
we can split up the data into two sets: a **training set** and a **test set**.
1. Learn $$Θ$$ and minimize $$J_{train}(\Theta)$$ using the training set
1. Compute the test set error $$J_{test}(\Theta)$$  

#### The test set error 
1. For linear regression: 
$$  J_{test}(\Theta) = \dfrac{1}{2m_{test}} \sum_{i=1}^{m_{test}}(h_\Theta(x^{(i)}_{test}) - y^{(i)}_{test})^2  $$  

1. For classification ~ Misclassification error (aka 0/1 misclassification error):  
$$  err(h_\Theta(x),y) =
\begin{matrix}
1 & \mbox{if } h_\Theta(x) \geq 0.5\ and\ y = 0\ or\ h_\Theta(x) < 0.5\ and\ y = 1\newline
0 & \mbox otherwise 
\end{matrix} $$  

The average test error for the test set is:  
$$  \text{Test Error} = \dfrac{1}{m_{test}} \sum^{m_{test}}_{i=1} err(h_\Theta(x^{(i)}_{test}), y^{(i)}_{test})  $$  

### Model Selection and Train/Validation/Test Sets  

**Without the Validation Set (note: this is a bad method - do not use it)**  
1. Optimize the parameters in $$theta$$ using the training set for each polynomial degree.  
   使用训练集优化不同多项式的参数$$theta$$
1. Find the polynomial degree d with the least error using the test set.
   利用测试集找出最小error的polynomial degree d  
1. Estimate the generalization error also using the test set with $$J_{test}(\Theta^{(d)})$$,(d = theta from polynomial with lower error);  
   评估。

In this case, we have trained one variable, d, or the degree of the polynomial, using the test set. This will cause our error value to be greater for any other set of data.  

**Use of the CV set** 使用交叉验证集  
To solve this, we can introduce a third set, the **Cross Validation Set**, to serve as an intermediate set that we can train d with. Then our test set will give us an accurate, non-optimistic error.
数据集小的话，用6-2-2的比例划分，如果数据集大的话，CV set 和 Test Set 不需要太大。  

**With the Validation Set (note: this method presumes we do not also use the CV set for regularization)**  
1. Optimize the parameters in $$theta$$ using the training set for each polynomial degree.  
1. Find the polynomial degree d with the least error using the cross validation set.
   利用交叉验证集找出最小error的polynomial degree d.    
1. Estimate the generalization error also using the test set with $$J_{test}(\Theta^{(d)})$$,(d = theta from polynomial with lower error);  
   评估。  
   
### Diagnosing Bias vs. Variance  
High bias (underfitting):both $$J_{train}(\Theta)$$ and $$J_{CV}(\Theta)$$ will be high. Also,$$J_{CV}(\Theta) \approx J_{train}(\Theta)$$J_{train}(\Theta)$$will be low and $$J_{CV}(\Theta)$$will be much greater than $$J_{train}(\Theta)$$  

### Regularization and Bias/Variance  
λ太大导致欠拟合， λ太小导致过拟合。  

### Learning Curves  
[Learning Curve](http://www.ritchieng.com/machinelearning-learning-curve/ "Learning Curve")  

## Machine Learning System Design  

------------
### Error Analysis  
The recommended approach to solving machine learning problems is:
1. Start with a simple algorithm, implement it quickly, and test it early.
1. Plot learning curves to decide if more data, more features, etc. will help
1. Error analysis: manually examine the errors on examples in the cross validation set and try to spot a trend.  

### Error Metrics for Skewed Classes  
**Precision**： （准不准） 

$$  \dfrac{\text{True Positives}}{\text{Total number of predicted positives}}
= \dfrac{\text{True Positives}}{\text{True Positives}+\text{False positives}}  $$  

**Recall**：  （全不全）  
$$  \dfrac{\text{True Positives}}{\text{Total number of actual positives}}= \dfrac{\text{True Positives}}{\text{True Positives}+\text{False negatives}}  $$  

**F Score **:  
$$  \text{F Score} = 2\dfrac{PR}{P + R} $$    

We want to train precision and recall on the cross validation set so as not to bias our test set.    


## Reference
1.[Machine Learning by Stanford University](https://www.coursera.org/learn/machine-learning/resources/LIZza)    
2.[Bias and Variance](http://www.cedar.buffalo.edu/~srihari/CSE555/Chap9.Part2.pdf)  
3.[managing-bias-variance-tradeoff-in-machine-learning](http://blog.stephenpurpura.com/post/13052575854/managing-bias-variance-tradeoff-in-machine-learning)




