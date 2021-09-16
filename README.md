# U-Net: Convolutional Networks for Biomedical Image Segmentation #

## Abstract ##

In this project, we have implemented the neural network architecture **U-Net** proposed in the paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf) to perform semantic segmentation. The architecture consists of a contracting path and a symmetric expansive path along with bridge connections. The authors promote use of data augmentation techniques specially *elastic deformation* to increase number of training samples and facilitate U-Net to train from few originally annotated training samples. The original implementation of U-Net was done in Caffe. We have implemented our version of U-Net in Python using Pytorch and trained it to segment membranes of neuronal structures.

## Combination of Affine and Elastic transformations/deformations to increase number of training samples ##
![alt text](https://github.com/nitinkmittal/biomedical_image_segmentation/blob/master/images/Original%20and%20Elastic%20Deformed.jpeg)