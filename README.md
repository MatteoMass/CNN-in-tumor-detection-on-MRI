# TumorDetector

This project was carried out for the Intelligent System for Pattern Recognition exam (6 credit version).

The aim of the project was to develop a Convolutional Neural Netowork to perform the semantic segmentation of brain MR images. Specifically, the model has to take in input an image (on the left) and return a mask (on the right) representing the portion of the brain that is a tumor.

<p align="center">
  <img src="https://user-images.githubusercontent.com/48138368/164726570-b1468e7d-1992-4e97-87de-1c641186ae54.png" />
</p>

# Model

The architecture used in this project is called Fully Convolutional Network, it differs from the classical Convolutional NN by replacing the last dense layers with an expansive path (convolution, deconvolution and upsampling layers) that allows to reconstruct the original image and make a pixel by pixel classification.

Both the first proposed version of FCN and an evolution of it, called U-net, have been tested. For both architectures
