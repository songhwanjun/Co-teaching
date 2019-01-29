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
- To inject noisy labels on CIFAR-10 dataset, the true label *i* was flipped to the randomly chosen label *j* with a probability *tau*. That is, *tau* determines the degree of noiseness on dataset.
- A densely connected neural networks (L=40, k=12)([Huang et al./ 2017](http://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html)) was used to train the noisy CIFAR-10.
- For the performance comparison, we compared the test loss of *Co-teaching* with that of *Default*. *Defualt* trained the noisy CIFAR-10 without any processing for noisy labels.

## 3. Environment
- Python 3.6.4
- Tensorflow-gpu 1.8.0 (pip install tensorflow-gpu==1.8.0)
- Tensorpack (pip install tensorpack)

## 4. How to Run
- Algorithm parameters
   ```python
    -gpu_id: gpu number which you want to use.
    -method_name: method in {Default, Co-teaching}.
    -noise_rate: the rate which you want to corrupt.
    -log_dir: log directory to save the training/test error.
   ```
   
- Algorithm configuration
  - Training epochs: 100
  - Batch size: 128
  - Learning rate: 0.1 (divided 5 at the 50% and 75% of the total number of epochs)
  - Dataset: CIFAR-10
  - These configuration can be easily modified:
  ```python
   from autoaugment import CIFAR10Policy
   data = ImageFolder(rootdir, transform=transforms.Compose(
                           [transforms.RandomCrop(32, padding=4, fill=128), # gray fill value is important bc of the color operations
                            transforms.RandomHorizontalFlip(), CIFAR10Policy(), 
             transforms.ToTensor(), 
                            Cutout(n_holes=1, length=16), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
                            transforms.Normalize(...)]))
   loader = DataLoader(data, ...)
   ```
  
  
- Running commend
   ```python
    python main.py gpu_id method_name noise_rate log_dir
   ```
   This commend includes:
   *i)* CIFAR-10 automatical download, 
   *ii)* noise injection, 
   *iii)* neural network training.

## 5. Tutorial (Simple Experiment)

  
  

 
