---
title: Distributed Tensorflow
date: 2018-02-05
categories:
- Machine Learning
- Tensorflow
tags: 
- Deep Learning
- Machine Learning 
- Tensorflow
description: Note of distributed tensorflow.
mathjax: true
---
# Distributed Tensorflow  

## Create a cluster  
A TensorFlow "cluster" is a set of "tasks" that participate in the distributed execution of a TensorFlow graph. Each task is associated with a TensorFlow "server", which contains a "master" that can be used to create sessions, and a "worker" that executes operations in the graph. A cluster can also be divided into one or more "jobs", where each job contains one or more tasks.  

### Create a tf.train.ClusterSpec to describe the cluster  
Create a**tf.train.ClusterSpec** that describes all of the tasks in the cluster. This should be the same for each task.

```python
tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
#job:local/task:0
#job:local/task:1

tf.train.ClusterSpec({
    "worker": [
        "worker0.example.com:2222",            #job:worker/task:0
        "worker1.example.com:2222",            #job:worker/task:1
        "worker2.example.com:2222"            #job:worker/task:2
    ],
    "ps": [
        "ps0.example.com:2222",         #job:ps/task:0
        "ps1.example.com:2222"         #job:ps/task:1
    ]})
```  

### Create a tf.train.Server instance in each task  
A **tf.train.Server** object contains a set of local devices, a set of connections to other tasks in its tf.train.ClusterSpec, and a **tf.Session** that can use these to perform a distributed computation. Each server is a member of a specific named job and has a task index within that job. A server can communicate with any other server in the cluster.  

```python
#  to launch a cluster with two servers running on localhost:2222 and localhost:2223
# run the following snippets in two different processes on the local machine

# In task 0:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=0)

# In task 1:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=1)
```  

## Specifying distributed devices in your model  
To place operations on a particular process, you can use the same tf.device function that is used to specify whether ops run on the CPU or GPU.  

```python
with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(...)
  biases_1 = tf.Variable(...)

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(...)
  biases_2 = tf.Variable(...)

with tf.device("/job:worker/task:7"):
  input, labels = ...
  layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
  logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
  # ...
  train_op = ...

with tf.Session("grpc://worker7.example.com:2222") as sess:
  for _ in range(10000):
    sess.run(train_op)
```  
**from ps to worker for the forward pass, and from worker to ps for applying gradients**    

## Replicated training  
A common training configuration, called "data parallelism," involves multiple tasks in a worker job training the same model on different mini-batches of data, updating shared parameters hosted in one or more tasks in a ps job. All tasks typically run on different machines. There are many ways to specify this structure in TensorFlow, and we are building libraries that will simplify the work of specifying a replicated model. Possible approaches include:  

- **In-graph replication**. In this approach, the client builds a single tf.Graph that contains one set of parameters (in tf.Variable nodes pinned to /job:ps); and multiple copies of the compute-intensive part of the model, each pinned to a different task in /job:worker.  

- **Between-graph replication**. In this approach, there is a separate client for each /job:worker task, typically in the same process as the worker task. Each client builds a similar graph containing the parameters (pinned to /job:ps as before using tf.train.replica_device_setter to map them deterministically to the same tasks); and a single copy of the compute-intensive part of the model, pinned to the local task in /job:worker.
