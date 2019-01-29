# Co-teaching - Robust training of deep neural networks with extremely noisy labels
Unofficial implementation of the CIFAR-10 learned by [Co-teaching](http://papers.nips.cc/paper/8072-co-teaching-robust-training-of-deep-neural-networks-with-extremely-noisy-labels).

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
To inject noisy labels on CIFAR-10 dataset, the true label *i* was flipped to the randomly chosen label *j* with a probability *tau*. 
A densely connected neural networks (L=40, k=12)([Huang et al./ 2017](http://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html)) was used to train the noisy CIFAR-10.

## 3. Example
