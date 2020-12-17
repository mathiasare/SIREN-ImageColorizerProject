---
layout: default
---


[Link to another page](./another-page.html).

# INTRODUCTION
As image colorization gives old photos a new life and dimension, it is something that has always compelled us. Hence assigning colors to black and white images is very bothersome, data scientist have come up with a solution to make it happen more easily by using artificial neural networks. Now, we want to take it to next level (more efficent and accurate) by using a recently composed and published SIREN activation function. The SIREN will be more thoroughly discussed in the next section.


The grand aim of this project is to build or reuse an image colorization neural network while using SIREN activation function instead.
SIREN is a periodic activation function described in [^sitz]. It has proven better results in fitting images, videos, audio signal, solving Poission equations etc than other existing activation functions. So, the question rose- can we use it in a neural network based model to colorize images. Furthermore, not only to use  it but will it perform better then for example much exploited _relu_ activation function.


The project is divided into three phases:
1. Experimenting with SIREN in the given Collab environment and getting acqainted with the concept.
2. Trying out b/w image colorization, merging and resizing.
3. **Building the SIREN based NN to colorize images**

The original SIREN paper github link: [SIREN](https://github.com/vsitzmann/siren)

# BACKGROUND

SIREN models use sinus as activation functions. Models build with SIREN are stable through gradient operation: if a model is trained an output, then the gradient of the output is also close from the aimed output.

This means that SIREN models trained to output an image will provide an output that also has a gradient -and laplacian- close to the aimed image.

![SIREN_gradient](imgs/SIREN_gradient_stable.png)

This stability of SIREN models is useful in performing operations on the output of SIREN models: the Fourier transform is compatible with SIREN models (it can be applied to the output of SIREN models), differential equations can be solved with SIREN because the derivative of functions are computed correctly through SIREN, ...



The fundamental use of SIREN models is to store images: a model is trained to output the value of the image on one pixel from the coordinates of the pixel as input. This means that the image is stored within the model, and can be resized at will. One interesting application is to merge images together: a model is trained to output an image which gradient will be the mean of 2 images. In the paper, this experiment was successful on gray images.

![SIREN_gradient](imgs/SIREN_merging_gray.png)

The resulting image is realistic, since it focuses on the main elements of each image.


Another application of SIREN models is the representation of shapes: the model will store a specific shape, trying to be as realistic as possible.

![SIREN_gradient](imgs/SIREN_shape.png)

We notice that SIREN results in a smoother result than other activation functions.

[^sitz]: Sitzmann, Vincent and Martel, Julien N.P. and Bergman, Alexander W. and Lindell, David B. and Wetzstein, Gordon, Implicit Neural Representations with Periodic Activation Functions, https://arxiv.org/pdf/2006.09661.pdf, 2020

# RESULTS

## Reusing model

### First results without SIREN

After messing around with the coloring model from [emil_wallner](https://medium.com/@emilwallner/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d) we eventually achieved something:
- 1000 epochs with only one picture to train and test
- Train photo, test photo, result

![Train photo](pics/img_0.png)
![Test photo](pics/test_0.jpg)
![Result](pics/res_0.jpg)

_in wish to make the model run in one's computer use tensorflow 1.14.0 and keras 2.1.6_
*(!pip install tensorflow==1.14.0)* *(!pip install keras==2.1.6)* *h5py=2.10.0*

### EMIL_WALLNER model

Many different combinations of activation functions were tried out but eventually three were chosen to carry out final experiments:
Let's call the models followingly:
- FULL_RELU(FR): everything is left as in the original model- only the ulitmate layer activation function is *tanh*, others are *relu*
- SIREN_RELU_SIREN(SRS): only the first and last layer are changed to have *siren* activation function
- SIREN130(S130) aka FULL_SIREN: Not only is it only using all *siren*, the *omega_0* is also chosen to be 1/30
All the mentioned models proved rather satisfing results on fitting one picture. The results of fitting are represented below:

![FR](pics/gifs/relufit.gif)
![SRS](pics/gifs/srsfit.gif)
![S130](pics/gifs/siren130fit.gif)

Final results after 1000 epochs

![FR_only](pics/gifs/relu_fit_only.png)
![SRS_fit](pics/gifs/srs_fit_only.png)
![S130only](pics/gifs/siren130_fit_only.png)

Loss over 1000 epochs:

![relu_loss](pics/gifs/relu_loss.PNG)
![SRS_fit](pics/gifs/srs_loss.PNG)
![S130only](pics/gifs/siren130_fit_only.png)

### First results with SIREN models: basic image representation

#### Steps of transformation

We consider image A and image B, we want to obtain image C as a mix of the 2 initial images. If we merge the images pixel by pixel, we will obtain new colors which we do not want. instead, we want only the main elements of each image to be present. Therefore, we are going to merge the gradients of the 2 images, and then build the image associated to this gradient. This operation does not require the use of SIREN networks. However, it is possible to use them as a storage format of each image.

![Model](SIREN_merging.png)

#### First test: merging 2 black and white images

We take 2 images 128*128 in black and white and merge them together. The results for each image are:
- image outputted by SIREN model
- gradient off the output
- laplacian of the output

The images we show are image A, B, and C.

![image_A_bw](imgs/image_A_bw.png)
![image_B_bw](imgs/image_B_bw.png)
![image_C_bw](imgs/image_C_bw.png)


We see that the resulting image has the details of both images. Since the elements are in the same positions in each image, it results is transparent shapes.

#### Second test: merging 2 RGB images

This step required creating the gradients of the functions by hand: for computational efficiency, we only computed the horizontal gradient of each image. We obtained the merged image as follows:

![image_A_rgb](imgs/image_A_rgb.png)
![image_B_rgb](imgs/image_B_rgb.png)
![image_C_rgb](imgs/image_C_rgb.png)


