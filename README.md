# Co-teaching - Robust training of deep neural networks with extremely noisy labels
Unofficial tensorflow implementation of the CIFAR-10 learned by [Co-teaching](http://papers.nips.cc/paper/8072-co-teaching-robust-training-of-deep-neural-networks-with-extremely-noisy-labels).

> __Publication__ </br>
> Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., Tsang, I.,
and Sugiyama, M. Co-teaching: Robust training of deep
neural networks with extremely noisy labels," *Advances in Neural Information Processing Systems (NIPS)*, pp.
8536–8546, 2018.

## 1. Summary
For robust training on noisy labels, *Co-teaching* uses two neural networks. Each network selects its small-loss samples as clean samples, and feeds such clean samples to its peer network for futher training. Below figure demonstrates the overall procedures of *Co-teaching*. For each iteration, two networks forward-propagate the same mini-batch to identify clean samples. Then, each selected clean subset is back-propagated into peer network to update the model parameter.
<p align="center">
<img src="figures/overview.png " width="650"> 
</p>

## 2. Noise Injection and Network Architecture
- To inject noisy labels on CIFAR-10 dataset, the true label *i* was flipped to the randomly chosen label *j* with a probability *tau*. That is, *tau* is a given noise rate that determines the degree of noiseness on dataset.
- A densely connected neural networks (L=40, k=12)([Huang et al./ 2017](http://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html)) was used to train the noisy CIFAR-10.
- For the performance comparison, we compared the test loss of *Co-teaching* with that of *Default*. *Defualt* trained the noisy CIFAR-10 without any processing for noisy labels.

## 3. Environment
- Python 3.6.4
- Tensorflow-gpu 1.8.0 (pip install tensorflow-gpu==1.8.0)
- Tensorpack (pip install tensorpack)

## 4. How to Run
- Algorithm parameters
   ```
    -gpu_id: gpu number which you want to use.
    -method_name: method in {Default, Co-teaching}.
    -noise_rate: the rate which you want to corrupt.
    -log_dir: log directory to save the training/test error.
   ```
   
- Algorithm configuration
   
   Data augmentation and distortion are not applied, and training paramters are set to:
   ```
   Training epochs: 200
   Batch size: 128
   Learning rate: 0.1 (divided 5 at the approximately 50% and approximately 75% of the total number of epochs)
   Dataset: CIFAR-10
   ```
   These configurations can be easily modified at main.py:
   ```python
   # gradient optimizer type
   optimizer = 'momentum'
   
   # total number of training epcohs
   total_epochs = 200
   
   # batch size
   batch_size = 128
   
   # learning rates used for training, and the time to use each learning rate.
   # e.g., lr=0.1 is used before 20,000 iterations, lr=0.02 is used before 30,000 iterations, lr=0.04 is used after 30,000 iterations
   lr_values = [0.1, 0.02, 0.004]
   lr_boundaries = [40000, 60000]
   
   # training algorithms
   if method_name == "Default":
       default(gpu_id, input_reader, total_epochs, batch_size, lr_boundaries, lr_values, optimizer, noise_rate, log_dir=log_dir)
   elif method_name == "Coteaching":
       coteaching(gpu_id, input_reader, total_epochs, batch_size, lr_boundaries, lr_values, optimizer, noise_rate, log_dir=log_dir)
   ```
  
   
- Running commend
   ```python
    python main.py gpu_id method_name noise_rate log_dir
   ```
   This commend includes:
   *i)* CIFAR-10 automatical download, 
   *ii)* noise injection, 
   *iii)* neural network training.

## 5. Tutorial 1: Comparison of learning curves at the noise rate of 40%.
- We set *tau* to 0.4
- Running script
   ```shell
   #!/bin/sh
   main.py 0 Default 0.4 tutorial_1/Defulat
   main.py 0 Co-teaching 0.4 tutorial_1/Co-teaching
   ```
- Running result
<p align="center">
<img src="figures/tutorial_1(1).png " width="650"> 
</p>


## 6. Tutorial 2: Comparison of the best test error with varying noise rates.
- We used *tau* in {0.0, 0.1, 0.2, 0.3, 0.4} //from *light* noise to *heavy* noise
- Running script
   ```shell
   #!/bin/sh
   for i in 0.0 0.1 0.2 0.3 0.4
   do
     main.py 0 Default $i tutorial_2/Defulat/$i
     main.py 0 Co-teaching $i tutorial_2/Co-teaching/$i
   done
   ```
- Running result
<p align="center">
<img src="figures/tutorial_2(1).png " width="400"> 
</p>

 
