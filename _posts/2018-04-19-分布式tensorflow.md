---
title: 分布式Tensorflow：在复杂多样的分布式系统中进行大规模机器学习
date: 2018-04-19
categories:
- Machine Learning
- Tensorflow
tags: 
- Deep Learning
- Machine Learning 
- Tensorflow
description: The white paper of Google's distributed tensorflow.
mathjax: true
---
  
## Introduction  
Tensorflow最早是由Google Brain Team开发，是一个使用data flow graphs的开源机器学习软件库，支持desktop， server或者带有单独API的移动端的多CPU或多GPU。  

## Programming Model and Basic Concepts  
> A TensorFlow computation is described by a directed graph, which is composed of a set of nodes. The graph represents a dataflow computation, with extensions for allowing some kinds of nodes to maintain and update persistent state and for branching and looping control structures within the graph in a manner similar to Naiad. Clients typically construct a computational graph using one of the supported frontend languages (C++ or Python). An example fragment to construct and then execute a TensorFlow graph using the Python front end is shown in Figure 1, and the resulting computation graph in Figure 2.  

<div  align="center">
<img src="http://p3ny2xk3h.bkt.clouddn.com/dtf1.png" style="width:600px;height:400px;">
</div>   

<div  align="center">
<img src="http://p3ny2xk3h.bkt.clouddn.com/dtf2.png" style="width:600px;height:400px;">
</div> 


### Operations and Kernels    
> An operation has a name and represents an abstract computation
(e.g., “matrix multiply”, or “add”). An operation can have attributes, and all attributes must be provided or inferred at graph-construction time in order to instantiate a node to perform the operation. One common use of attributes is to make operations polymorphic over different tensor element types (e.g., add of two tensors of type float versus add of two tensors of type int32). A kernel is a particular implementation of an operation that can be run on a particular type of device (e.g., CPU or GPU). A TensorFlow binary defines the sets of operations and kernels available via a registration mechanism, and this set can be extended by linking in additional operation and/or kernel definitions/registrations. Table 1 shows some of the kinds of operations built into the core TensorFlow library.  

一个操作有一个名字。它表示一个抽象的计算（比如说，“矩阵相乘”或者“相加”）。一个操作可以有属性（attribute），所有的属性必须提供或者在图构造的过程中推断出以实例化一个节点来执行操作。属性通常的使用方式是让操作在不同的张量元素类型上多态（例如，两个 float 类型的张量和两个 int32 类型的张量）。核（kernel）是一种操作的特别实现，可以运行在一个特定类型的设备上（如 CPU 或者 GPU）。TensorFlow 的 binary 定义了可以通过注册（registration）机制实现的操作和核的集合，这个集合可以通过连接额外的操作/核的定义/注册。表 1 展示了内置于 TensorFlow 核心库的一些操作类型。  

<div  align="center">
<img src="http://p3ny2xk3h.bkt.clouddn.com/dtf3.png" style="width:800px;height:400px;">
</div> 

### Sessions
> Clients programs interact with the TensorFlow system by creating a Session. To create a computation graph, the Session interface supports an Extend method to augment the current graph managed by the session with additional nodes and edges (the initial graph when a session is created is empty). The other primary operation supported by the session interface is Run, which takes a set of output names that need to be computed, as well as an optional set of tensors to be fed into the graph in place of certain outputs of nodes. Using the arguments to Run, the TensorFlow implementation can compute the transitive closure of all nodes that must be executed in order to compute the outputs that were requested, and can then arrange to execute the appropriate nodes in an order that respects their dependencies (as described in more detail in 3.1). Most of our uses of TensorFlow set up a Session with a graph once, and then execute the full graph or a few distinct subgraphs thousands or millions of times via Run calls.     

客户端通过创建会话（session）和 TensorFlow 系统进行交互。为了创建一个计算图，会话接口支持外部（external）方法来提升当前由包含额外节点和边的会话的图（当会话创建时初始的图是空的）。另一个由会话接口提供的主要的操作就是 Run，以需要计算的输出名称和替换某些输出节点的张量的操作集合作为其参数输入。通过控制 Run 的参数，TensorFlow 的实现可以计算所有节点的必须执行传递闭包来计算需要的输出，然后安排执行合适节点来保证他们的依赖关系。大多数 TensorFlow 的使用都是针对一个图启动一个会话，然后执行整个图或者通过 Run 调用来执行分离的子图数千或者数百万次。  

### Variables  
在大多数计算中，图都是执行多次的。大多数的张量在一次执行后不会存活。然而，变量（variable）是一种特别的操作可以返回一个在图执行若干次过程中存活的持久化的可变张量的句柄。这个句柄可以传递给一系列特定的操作，例如 Assign 和 AssignAdd（等同于 +=）就可以改变其引用的张量了。对应 TensorFlow 在机器学习中的应用，模型的参数典型地就存放在变量引用的张量中，并作为模型训练图的 Run 的一部分进行更新。

