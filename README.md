# ann
ANN is short for artifcial neural network; it's widely used for pattern recognition (including image recogintion, motion dection, etc). This is just a start for pratical application of ANN. Beside the paralelle computation, ANN can be simplified into maxtrix compuation. This project is to split the computation into small computing units - neuron. This will have better performance for ANN with thousands of neurons in it.
# Parallelizations
There are two effective approaches to parallel network training:
* Data Parallelizations: The training data is devided into disjoint set. Each thread has its own network and works on it own dataset. Weights synchronization occurs periodically when N frames are processed.
* Node Parallelizations: In this case, there is only one instance of the network. The network layers are divided into disjoint
sets of neurons. Each thread has associated its own set.This method imposes higher frequency of synchronization than data parallelization.
* Approach in this project is similar to second one but it's DIFFERENT: each thread has NO associated its own set. It's more like a threadpool consuming network nodes in a thread-safety queue. Neurons in lower layers are always pushed into the queue than those in higher layers. The target is to deminlate the overhead for high-frequency-synchronization.
# Status
This project is still in progress. First batch of test results are achieved. More specifically, the status is updated in TO-DO section.
# Test Results
## MNIST
Four files need to be downloaded from http://yann.lecun.com/exdb/mnist/.
* train-images-idx3-ubyte
* train-labels-idx1-ubyte
* t10k-images-idx3-ubyte
* t10k-labels-idx1-ubyte
First two is for training and others for testing.
## Test environments:
* Memory: 2.5 GiB
* Processor: Genuine Intel® CPU T2500 @ 2.00GHz × 2
* OS Type: Ubuntu 16.04, 32-bit

|       Dataset      |Number of Samples|Iteration |Number of Neurons|Number of Threads|Training time:(s)|Accuracy Ratio (%)|
|--------------------|-----------------|----------|-----------------|-----------------|-----------------|------------------|
|               MNIST|     60000(10000)|         1|    784x100x10   |                4|          147.619|             94.64|
|               MNIST|     60000(10000)|         1|784x256x256x10   |                4|          378.866|             94.63|

# To-Do
|                    Task                     |   Status  |      Date     |                       Comment                   |  
|---------------------------------------------|-----------|---------------|-------------------------------------------------|
|       thread-pool support for large networks|   Done    |    Dec 27     |Function test is done for Logic And; TODO: MNIST |
|                        performance benchmark|In progress|    Dec 31     |Test is done for MNIST                           |                             
|                                update Readme|In Progress|    Dec 31     |Updated                                          |
|applied to MNIST (Handwriting digit data set)|       Done|    Dec 31     |UT and other function test are broken; added new Task                                                 |
|                           More test on MNIST|        New|               |                                                 |
|              Fix broken UT and function test|        New|               |                                                 |
