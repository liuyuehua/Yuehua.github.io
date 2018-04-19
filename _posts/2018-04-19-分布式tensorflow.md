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

## Implementation  
TensorFlow 系统的主要部分就是客户端，它使用了会话接口来和 master 及一个或者多个的 worker processes 进行通信，每个 worker process 负责对一个或者多个计算设备（CPU 核或者 GPU card）的任意访问和在这些设备上进行图节点的计算按照 master 的要求执行。我们有本地和分布式实现的 TensorFlow 接口。本地实现通常是客户端、master 和 worker 都是在同一台机器上在一个单一的操作系统进程（可能包括多个设备，比如说装了多个 GPU  card的设备）上运行。分布式实现采用了本地实现的很多的代码，但是扩展了对客户端、master 和 worker 可以在不同的机器的不同的进程上运行的场景支持。在我们的分布式环境中，这些不同的任务对应于 cluster 调度系统分配在 job 中的容器中。这两种不同的模式在图 3 中进行的展示。  

<div  align="center">
<img src="http://p3ny2xk3h.bkt.clouddn.com/dtf4.png" style="width:600px;height:400px;">
</div> 

#### Devices  
设备是 TensorFlow 的计算核心。每个 worker 负责一个或者多个设备，每个设备有一个设备类型和一个名字。设备名字由识别设备类型的部分，在 worker 中的设备索引，以及在分布式设定中，worker 的 job和任务（或者 localhost 当设备是和进程在同一机器时）的标志构成。一些例子如/job: localhost/ device : cpu:0 或者 /job :worker/ task: 17/ device : gpu:3。 我们已实现了 CPU 和 GPU 的设备接口而其他的设备类型也有了通过注册机制完成的设备实现方式。每个设备对象负责管理分配和解除分配设备内存，对在 TensorFlow 实现中的更高层请求任意 kernel 的执行调度管理。  

#### Tensors  
> A tensor in our implementation is a typed, multidimensional array. We support a variety of tensor element types, including signed and unsigned integers ranging in size from 8 bits to 64 bits, IEEE float and double types, a complex number type, and a string type (an arbitrary byte array). Backing store of the appropriate size is managed by an allocator that is specific to the device on which the tensor resides. Tensor backing store buffers are reference counted and are deallocated when no references remain.  

实现中的张量是一种有类型的、多维度数组。我们支持若干张量元素类型，包含大小为从 8 bit 到 64 bit 的带符号和无符号整型，IEEE 浮点数和双精度类型、复数类型和字符串类型（任意长的字节数组）。合适大小的后台存储通过一个分配器进行管理，该分配器由张量所处的设备确定。张量的后端存储缓存是引用计数的并在没有引用存在时解除分配。

### Single-Device Execution  
> Let’s first consider the simplest execution scenario: a single worker process with a single device. The nodes of the graph are executed in an order that respects the dependencies between nodes. In particular, we keep track of a count per node of the number of dependencies of that node that have not yet been executed. Once this count drops to zero, the node is eligible for execution and is added to a ready queue. The ready queue is processed in some unspecified order, delegating execution of the kernel for a node to the device object. When a node has finished executing, the counts of all nodes that depend on the completed node are decremented.  

首先考虑最简单的执行场景：单一的worker进程运行在单一的设备上。图上的节点按照节点之间的依赖顺序执行。特别地，我们会在每个节点上保持一个计数来记录这个节点上还没有执行的依赖。一旦这个计数变为 0，节点就可以被调度使用，并会加入到待续的队列中。待续队列按照某个非指定的顺序处理，指派节点执行的kernel 到设备对象上。当一个节点完成执行，所有依赖这个完成的节点的节点的计数都会减少。

### Multi-Device Execution  
一旦系统有了多个设备，有两个主要的复杂情形出现：确定图中每个节点的计算所处的设备，然后管理由上一步确定的置放决定所产生的设备间所需的数据通信。后续部分讨论这两个问题。 

#### Node Placement  
> The placement algorithm first runs a simulated execution of the graph. The simulation is described below and ends up picking a device for each node in the graph using greedy heuristics. The node to device placement generated by this simulation is also used as the placement for the real execution.  

该置放算法的输入是一个代价模型，包括对每个图节点的输入和输出张量的规模的估计，和对每个节点在给与其输入张量时的计算时间的评估。这个代价模型或者是基于关联不同操作类型的启发式规则的静态估计，或者基于实际的为了图能够更早执行而做的置放决定集合的衡量。

置放算法首先运行模拟的图的执行过程。模拟按照下面描述进行，最终对每个节点使用贪心策略选择一个设备。节点到设备的置放过程也是用作真实执行的置放。

置放算法从计算图的源点开始，在系统中的每个设备上模拟相应的活动。对每个在遍历中抵达的节点，可选 available 设备的集合会被考虑到（设备可能会由于其没能提供实现了特定操作的kernel而不可选）。对那些拥有多个可选设备的节点，置放算法使用一种贪心策略来检查在每个可能的置放节点上需要的完成时间的效果来决定置放决策。这种启发式规则考虑了根据代价模型在那种设备上估计的和衡量的执行时间，还有任何用来从其他设备传输输入到该节点的通信的代价。其中节点的操作完成最快的设备会被选作该操作的设备，置放决策然后会继续针对图中其他的节点进行处理，包含那些已经做好模拟执行的下游节点。**第 4.3 节描述了一些扩展，让用户可以提供提示和部分限制来指导置放算法。这个算法现在还在持续开发的过程中。**  

#### Cross-Device Communication  
一旦节点置放已经计算好，图就被划分成子图的集合，每个子图对应于一个设备。从 x到 y 任何交叉设备的边都会被移除并用一条从 x 到一个 x 的子图中新的 Send 节点的边和从在 y 子图中对应的 Receive 节点到 y 的边代替。参见图 4 中所进行的变换。  

<div  align="center">
<img src="http://p3ny2xk3h.bkt.clouddn.com/dtf5.png" style="width:500px;height:300px;">
</div>  

在运行时刻，Send 和 Receive 节点合作进行跨设备的数据交换。这使得我们可以隔离所有在 Send 和 Receive 内部实现的通信，这样简化了运行时刻剩下的部分工作。
当我们插入 Send 和 Receive 节点时，我们将在特定设备上的特定张量的所有使用者进行合并规整来使用单个 Receive 节点，而不是对特定设备上的每个下游使用者都给一个 Receive 节点。这确保了需要使用的张量数据仅仅会从源设备到目的设备传输一次，而在目的设备上的张量内存也只会分配一次（而非多次，参看图 4 的节点 b 和 c）。

通过这种方式处理通信，我们也允许了不同设备上的图中的个别节点调度可以被去中心化到 workers 上：Send 和 Receive 节点传达了在不同的 worker 和 设备间必要的同步信息，master 仅仅需要对每个图的执行给出一个 Run 请求给那些包含图中任意节点的 worker，而不是会对所有节点或者每个跨设备通信都进行调度。这也让系统更加可扩展，并允许比通过 master 来强制进行所有的调度更加精确的节点执行。

### Distributed Executio  
> Distributed execution of a graph is very similar to multidevice execution. After device placement, a subgraph is created per device. Send/Receive node pairs that communicate across worker processes use remote communication mechanisms such as TCP or RDMA to move data across machine boundaries.

计算图的分布式执行非常类似于多设备执行。在设备置放后，子图会针对每个设备创建。用于 worker 进程之间的通信的 Send/Receive 节点对使用了诸如 TCP 或者 RDMA 这样的远程通信机制进行跨机器的数据迁移。 

#### Fault Tolerance  
分布式执行中的错误可以在很多地方进行检测。最主要的有 (a) 在 Send 和 Receive 节点对之间的通信错误，(b) 从 master 进程到每个 worker 进程的周期性的健康状态检测。

如果发现了错误，整个图的执行就会终止，并从头开始。但是回想之前变量节点对应于那些在执行过程中记忆持有（persist）的张量（Recall however that Variable nodes refer to tensors that persist across executions of the graph.）。我们支持在重启过程中的一致的检查点和状态恢复。特别是，每个变量节点连接在一个 Save 节点上。这些 Save 节点周期性地执行，比如说每 N 次迭代，或者每隔 N 秒。他们执行的时候，变量的内容被写到持久化的存储中，比如说，一个分布式的文件系统。类似地，每个变量连接在一个 Restore 节点上，只会在一次重启后的第一个迭代中启用。在 4.2 节有某些节点仅能够在某些图的执行中启用的细节。