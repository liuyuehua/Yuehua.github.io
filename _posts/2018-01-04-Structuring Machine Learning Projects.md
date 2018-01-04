---
title: Structuring Machine Learning Projects
date: 2018-01-04
categories:
- Deep Learning
tags: 
- Deep Learning
- Machine Learning 
- deeplearning.ai
description: How to build a successful machine learning project..
mathjax: true
---
### Orthogonalization  
Orthogonalization or orthogonality is a system design property that assures that modifying an instruction or a component of an algorithm will not create or propagate side effects to other components of the system. It becomes easier to verify the algorithms independently from one another, it reduces testing and development time.  
对于需要调整什么来达到某个效果，这步骤叫做**正交化**.  

如果根据某个Cost Function， 系统用在测试集上很好，但无法反应该算法在真实数据中的表现，这意味着要么开发集的分布设置不正确，要么是Cost Function的测量指标不对。  

在训练神经网络时， NG建议一般不要用early stopping。 因为一般情况下early stopping不太正交化。 单一影响的手段调网络会简单不少。

### Single number evaluation metric  
一个验证集合和单的数字评估指标可以加速改进机器学习算法迭代过程。  

### Satisficing and Optimizing metric  
Satisficing metric： 如运行时间，消耗内存等。满足指标只需达到设置的阈值即可。  

### Train/dev/test distributions  
Setting up the training, development and test sets have a huge impact on productivity. It is important to choose the development and test sets from the same distribution and it must be taken randomly from all the data.  
**Guideline**  
Choose a dev set and test set to reflect data you expect to get in the future and consider important to do well on.  

### Size of the dev and test sets  
**Modern era – Big data**
Now, because a large amount of data is available, we don’t have to compromised as much and can use a greater portion to train the model.  

**Guidelines**  
1. Set up the size of the test set to give a high confidence in the overall performance of the system.
1. Test set helps evaluate the performance of the final classifier which could be less 30% of the whole data set.
1. The development set has to be big enough to evaluate different ideas.

### When to change dev/test sets and metrics  
在当前dev set或者test set中表现很好，但在实际应用中表现不好时， 需要修改metric或者dev set.  

### human-level performance  
贝叶斯最优误差：性能无法超过某个理论上限。  

### Avoidable bias
可避免误差是贝叶斯和训练之间的差值，理论上讲训练误差无法比贝叶斯误差好，除非过拟合。  

### Improving your model performance  
**The two fundamental assumptions of supervised learning**  

There are 2 fundamental assumptions of supervised learning. The first one is to have a low avoidable bias which means that the training set fits well. The second one is to have a low or acceptable variance which means that the training set performance generalizes well to the development set and test set.  

If the difference between human-level error and the training error is bigger than the difference between the training error and the development error, the focus should be on bias reduction technique which are training a bigger model, training longer or change the neural networks architecture or try various hyperparameters search.  

If the difference between training error and the development error is bigger than the difference between the human-level error and the training error, the focus should be on variance reduction technique which are bigger data set, regularization or change the neural networks architecture or try various hyperparameters search.

### Carrying out error analysis  
误差分析： 找一组错误例子（可能在dev或者test）， 观察错误标记的例子，统计并归纳。  

### Cleaning up incorrectly labeled data  
事实证明深度学习算法对于中的随机误差是相当鲁棒的。但对系统性的错误就没那么好的鲁棒。  

### Build your first system quickly, then iterate

### Training and testing on different distributions  
需求目标最大。  

### Bias and Variance with mismatched data distributions  
- 算法只见过训练集数据,没见过开发集数据。  
- 开发集数据来自不同的分布。    
- 需要辨清开发集上的误差有多少是因为算法没看到开发集中的数据导致的(*方差*),多少是因为开发集数据分布本身就不一样(*数据不匹配*).  

**Solution**  
- 定义一个新的数据train-dev set 从训练集中抽取数据,和训练集数据来自同一个数据分布,但是不用于训练数据.  
- 分别将分类器在训练集/训练-开发集/开发集上运行,获取其准确率信息  
- 假如在训练集上误差为1%,在训练-开发集上误差为9%,在开发集上误差为10%  
分类器在训练集和训练开发集上误差差距较大,这表明算法没有识别没有看到过的数据,这表明分类器本身**方差较大**  
分类器在训练-开发集和开发集上误差差距不大,表明算法误差的差距不是主要由于数据 **分布不一致** 导致的  
- 假如在训练集上误差为1%,在训练-开发集上误差为1.5%,在开发集上误差为10%  
分类器在训练集和训练开发集上误差差距较小,这表明分类器本身**方差不大  **
分类器在训练-开发集和开发集上误差差距很大,表明算法误差的差距主要由于 **数据不匹配** 导致的  

### Addressing data mismatch  
- 误差分析  
- 谨慎使用人工数据合成  

### Transfer learning  
- Task A and B have the same input x.  
- You have a lot more data for Task A than Task B.  
- Low level features from A could be helpful for learning B.  

### Multi-task learning  
- Training on a set of tasks that could benefit from having shared lower-level features.  
- Usually: Amount of data you have for each task is quite similar.  
- Can train a big enough neural network to do well on all the tasks.  

### End-to-End Deep Learning  
简而言之，以前有一些数据处理系统或者学习他们需要多个阶段的，那么端到端深度学习就是忽略所有这些不同的阶段，用单个神经网代替它。  
需要大量的数据。  

### Reference
1.[Deep Learning](https://www.deeplearning.ai/)  
2.[Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/) 