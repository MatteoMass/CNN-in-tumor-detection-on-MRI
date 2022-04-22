# TumorDetector

This project was carried out for the <i>Intelligent System for Pattern Recognition</i> exam (6 credit version).

The aim of the project was to develop a Convolutional Neural Netowork to perform the semantic segmentation of brain MR images. Specifically, the model has to take in input an image (on the left) and return a mask (on the right) representing the portion of the brain that is a tumor.

<p align="center">
  <img src="https://user-images.githubusercontent.com/48138368/164726570-b1468e7d-1992-4e97-87de-1c641186ae54.png" />
</p>

# Model

The architecture used in this project is called <b>Fully Convolutional Network</b>, it differs from the classical Convolutional NN by replacing the last dense layers with an expansive path (convolution, deconvolution and upsampling layers) that allows to reconstruct the original image and make a pixel by pixel classification.

Both the first proposed version of FCN and an evolution of it, called <b>U-net</b>, have been tested. Different configurations were tested for both architectures in order to find the best performing combination of number of filters, loss function and dropout. The metrics used to evaluate the models are <i>Accuracy</i> and <i>Dice Coefficient</i>.

Particular attention was given to loss functions, and given the nature of the problem, several were tested:
- <i>Binary Crossentropy</i>
- <i>Dice Loss</i>
- <i>Tversky Loss</i>

# Result
The model that gave the best results was the U-net (image below) in combination with binary crossentropy.
<p align="center">
  <img src="https://user-images.githubusercontent.com/48138368/164784623-2336732f-b9c1-49d1-9cdd-54a4ff071095.png" />
</p>
This model managed to achieve a coefficient dice of 0.72 in the test set, and produced predictions such as those below where from left to right is the original image, ground truth and prediction. As can be seen, the model is able to identify the location, shape and size of the tumour quite well, although some false positives are present (as in the second set of images).

<p align="center">
  <img src="https://user-images.githubusercontent.com/48138368/164784824-3d5a54b9-a0d8-46e0-8b0f-d6f997d34fc5.png">
  <img src="https://user-images.githubusercontent.com/48138368/164784836-f25d8d3d-63d1-4499-9f64-583b2bafe56b.png">
</p>

More detailed information can be found in the <i>report.pdf</i> file in this repository.
