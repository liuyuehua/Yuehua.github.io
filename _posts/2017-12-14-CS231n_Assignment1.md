---
title: Image Classfication
date: 2017-12-14
categories:
- CNN for Visual Recognition 
tags: 
- CS231n
- Image Classfication
description: A simple image classification pipeline, based on the k-Nearest Neighbor or the SVM/Softmax classifier.
mathjax: true
---
## k-Nearest Neighbor Classifier
**L1 distance:**

$$ d_1 (I_1, I_2) = \sum_{p} \left| I^p_1 - I^p_2 \right| $$

**L2 distance:**

$$ d_2 (I_1, I_2) = \sqrt{\sum_{p} \left( I^p_1 - I^p_2 \right)^2} $$

```python
distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))

```

**L1 vs. L2.** It is interesting to consider differences between the two metrics. In particular, the L2 distance is much more unforgiving than the L1 distance when it comes to differences between two vectors. That is, the L2 distance prefers many medium disagreements to one big one. L1 and L2 distances (or equivalently the L1/L2 norms of the differences between a pair of images) are the most commonly used special cases of a p-norm. L2更不能容忍较大差异。

### A kNN classifier with L2 distance
```python
num_test = X.shape[0]
num_train = self.X_train.shape[0]
dists = np.zeros((num_test, num_train)

dists = np.sqrt(-2*np.dot(X, self.X_train.T) + np.sum(np.square(self.X_train), axis = 1) + np.transpose([np.sum(np.square(X), axis = 1)]))
```
[k_nearest_neighbor.py](https://github.com/lightaime/cs231n/blob/master/assignment1/cs231n/classifiers/k_nearest_neighbor.py "k_nearest_neighbor.py")

### Cross-validation
**用来训练超参数，例如K-nearest-neighbour classifier中的超参数K.**
In cases where the size of your training data (and therefore also the validation data) might be small, people sometimes use a more sophisticated technique for hyperparameter tuning called cross-validation. Working with our previous example, the idea is that instead of arbitrarily picking the first 1000 datapoints to be the validation set and rest training set, you can get a better and less noisy estimate of how well a certain value of k works by iterating over different validation sets and averaging the performance across these. For example, in 5-fold cross-validation, we would split the training data into 5 equal folds, use 4 of them for training, and 1 for validation. We would then iterate over which fold is the validation fold, evaluate the performance, and finally average the performance across the different folds.

## Linear SVM Classifier
### Multiclass Support Vector Machine loss

$$ L_i = \sum_{j\neq y_i} \max(0, s_j - s_{y_i} + \Delta) $$

$$ L_i = \sum_{j\neq y_i} \max(0, w_j^T x_i - w_{y_i}^T x_i + \Delta) $$

$$ L =  \underbrace{ \frac{1}{N} \sum_i L_i }_\text{data loss} + \underbrace{ \lambda R(W) }_\text{regularization loss} \\\\ $$

### Linear_SVM
```python
 loss = 0.0
 dW = np.zeros(W.shape) # initialize the gradient as zero
 #a vectorized version of the structured SVM loss
 num_train = X.shape[0]
 num_classes = W.shape[1]
 scores = X.dot(W)
 correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1) #(N, 1)
 margins = np.maximum(0, scores - correct_class_scores +1)
 margins[range(num_train), list(y)] = 0
 loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)

 # Implement a vectorized version of the gradient for the structured SVM     #
 # loss, storing the result in dW. 
 coeff_mat = np.zeros((num_train, num_classes))
 coeff_mat[margins > 0] = 1
 coeff_mat[range(num_train), list(y)] = 0
 coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

 dW = (X.T).dot(coeff_mat)
 dW = dW/num_train + reg*W
```
[linear_svm.py](https://github.com/lightaime/cs231n/blob/master/assignment1/cs231n/classifiers/linear_svm.py "linear_svm.py")

## Softmax classifier
### Softmax Loss function

$$ L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right) \hspace{0.5in} \text{or equivalently} \hspace{0.5in} L_i = -f_{y_i} + \log\sum_j e^{f_j} $$

### Softmax_classifier
```python
  loss = 0.0
  dW = np.zeros_like(W)
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)
  softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis =  1).reshape(-1,1)
  loss = -np.sum(np.log(softmax_output[range(num_train), list(y)]))
  loss /= num_train 
  loss +=  0.5* reg * np.sum(W * W)
  
  dS = softmax_output.copy()
  dS[range(num_train), list(y)] += -1
  dW = (X.T).dot(dS)
  dW = dW/num_train + reg* W 

```
[softmax.py](https://github.com/lightaime/cs231n/blob/master/assignment1/cs231n/classifiers/softmax.py "softmax.py")


## Reference
[CS231n: Convolutional Neural Networks for Visual Recognition by Stanford University](http://cs231n.stanford.edu/index.html)
[Image Classification](http://cs231n.github.io/classification/)
[Linear Classification](http://cs231n.github.io/linear-classify/)