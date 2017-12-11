---
title: Backpropagation and Neural Networks
date: 2017-12-11
categories:
- Deep Learning
tags: 
- Deep Learning
description: Table of Contents.
---

#Introduction
## Motivation
In this section we will develop expertise with an intuitive understanding of backpropagation, which is a way of computing gradients of expressions through recursive application of chain rule. Understanding of this process and its subtleties is critical for you to understand, and effectively develop, design and debug Neural Networks.
质能方程$$E = mc^2$$
$$X = A_{1}^N + A_{2}^N + A_{3}^N$$
$$f(x, y) = 100 * \lbrace[(x + y) * 3] - 5\rbrace$$
$$\sqrt[3]{X}$$
```python
import numpy as np 
import matplotlib.pyplot as plt 

greyhunds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhunds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()
```
