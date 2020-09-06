  

# Accelerating Training Using Multi GPUs

  

  

## Experimental

  

  

The Training Environment: ``Athena``

  

  

Training Data: A subset was random selected 1000 samples from HKUST training dataset.

  

  

Network: ``LAS`` Model

  

  

Primary Network Configuration: ``NUM_EPOCHS`` 1， ``BATCH_SIZE`` 10

  

  

The training time is changed by different number of server and GPU when using `Horovod`+`Tensorflow`. At the same time, the training data and network structure etc still keep same to train one epoch. These results of experiment are shown below:

  

  

### The training time using ``Horovod``+``Tensorflow``(Character)


Server and GPU number | 1S - 1GPU | 1S - 2GPU | 1S - 3GPU | 1S - 4GPU | 2S - 2GPU | 2S - 4GPU | 2S - 6GPU | 2S - 8GPU 
:-----------: | :------------: | :----------: |  -------: | -------: | :-----------: | :------------: | :----------: |  -------:
Training time(s/1 epoch) | 121.409 | 83.111 | 61.607 | 54.507 | 82.486 | 49.888 | 33.333 | 28.101 |

  

  

## The Result Analysis

  

1. As shown in Table above, training time gets shorter when more GPUs are used. The speedup using four GPU is 2.2 times compared to using one GPU.

  

  

2. The communication overhead is really small between difference server using `Horovod`. We have trained model with same structure respectively using 1 servers with 2 GPUs and using 2 servers with 1 GPU each. The total training time is almost the same.