# The efficiency of GPU Using '``Horovod``+``Tensorflow``'

## Experimental

The Training Environment: ``Athena``


Traning Data: A subset was random selected 1000 samples from HKST training dataset.



Newwork: ``LAS`` Model

Primary Network Configuration: ``NUM_EPOCHS`` 1， ``BATCH_SIZE`` 10


 
 The training time is changed by deferent number of of server and GPU when using `Horovod`+`Tensorflow`. As the same time, the training data and network structure etc still keep same to train `one` `epoch`. These results of experiment as follow:

### The training time using ``Horovod``+``Tensorflow``(Character)


Server and GPU number | 1S-1GPU | 1S-2GPUs | 1S-3GPUs | 1S-4GPUs | 2Ss-2GPUs | 2Ss-4GPUs |  2Ss-6GPUs |  2Ss-8GPUs |
:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
training time(s/1 epoch) | 121.409 | 83.111  | 61.607  |  54.507 |  82.486 | 49.888  | 33.333   |  28.101  |

## The Reslut Analysis
1. As the character shown that the more GPUs are used and the training time is shorter. For example, we commpared their training time scale between using 1 server with 1 GPU and 1 server with 4 GPUs. Their training time scale is `1S-4GPUs:1S-1GPU=1:2.22`. Moreover，anoter set of data is recorded as `2Ss-8GPUs:1S-1GPU=1:4.3`. From them we can see, increasing the number of GPU when we train model can save training time and increase the efficiency.

2. The communication time is really short between difference server using `Horovod`. We have trained the same structure model respectively using 1 servers with 2 GPUs and using 2 servers with 1 GPU each and the training time scale is `1S-2GPUs:2Ss-2GPUs=1:1`.

