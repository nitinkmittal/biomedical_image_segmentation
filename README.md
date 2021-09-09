# U-Net: Convolutional Networks for Biomedical ImageSegmentation #

This README would normally document whatever steps are necessary to get your application up and running.

## Abstract ##

In this project, we have implemented the neural network architecture U-Net proposed in the paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf) to perform semanticimage segmentation. The architecture consists of a contracting path and a symmetric expansive path along with bridge connections. The authors promote use of data augmentationto facilitate U-Net to train from few annotated training samples originally. The original implementation of U-Net was done in Caffe. We have implemented our version of U-Net inPython using Pytorch and trained it to segment membranes of neuronal structures.

![alt text](https://github.com/nitinkmittal/biomedical_image_segmentation/blob/master/images/Image%20and%20Predicted%20Mask.png)