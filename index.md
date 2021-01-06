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

However, the training of a model using sine activation functions depends highly on the initialization of the activation functions:
```
y=sin(ax+b), a, b \in \mathbb{R}
```
If the initialization is inaccurate, the model may fail at training. The advised strategy to initialize weights is:
```
a \tilde \mathbb{U}(-\dfrac{\sqrt{6}}{n}, \dfrac{\sqrt{6}}{n})
```
n is the number of nodes in the layer. This choice allows the result ax+b to be in $[-\dfrac{\pi}{2}, \dfrac{\pi}{2}]$, and have  a normal distribution. Therefore, the initialization avoids a situation in which a learning step jumps over entire periods of the sine function.


The fundamental use of SIREN models is to store images: a model is trained to output the value of the image on one pixel from the coordinates of the pixel as input. This means that the image is stored within the model, and can be resized at will. One interesting application is to merge images together: a model is trained to output an image which gradient will be the mean of 2 images. In the paper, this experiment was successful on gray images.

![SIREN_gradient](imgs/SIREN_merging_gray.png)

The resulting image is realistic, since it focuses on the main elements of each image.


Another application of SIREN models is the representation of shapes: the model will store a specific shape, trying to be as realistic as possible.

![SIREN_gradient](imgs/SIREN_shape.png)

We notice that SIREN results in a smoother result than other activation functions.

[^sitz]: Sitzmann, Vincent and Martel, Julien N.P. and Bergman, Alexander W. and Lindell, David B. and Wetzstein, Gordon, Implicit Neural Representations with Periodic Activation Functions, https://arxiv.org/pdf/2006.09661.pdf, 2020

# RESULTS

# Colorization

Here we represent what we achieved with colorization.

## Reusing model

What kind of model should we use? It was a rather complicated question at first and we concentrated too much on quite complicated models such as [AIC](https://github.com/lukemelas/Automatic-Image-Colorization) and [Pix2Pix](https://github.com/affinelayer/pix2pix-tensorflow). Then we found a simple introduction to image colorization given by Emil Wallner from [emil_wallner](https://medium.com/@emilwallner/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d). It also inculded advanced model that were capable of actually colorizing b/w images. It was based on _relu_ and _tanh_ activation functions. So, it was a perfect model to try out _siren_ function.

Here is the overview of the architecture of the model:

```
embed_input = Input(shape=(1000,))

#Encoder
encoder_input = Input(shape=(256, 256, 1,))
encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

#Fusion
fusion_output = RepeatVector(32 * 32)(embed_input)
fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
fusion_output = concatenate([encoder_output, fusion_output], axis=3)
fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)

#Decoder
decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)

model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
```

the `embed_input` includes information from the image classifier — the inception resnet v2, that is trained on over a million images. The more exact description of the model is given at [emil_wallner](https://medium.com/@emilwallner/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d).


### Warm-up

After messing around with the [emil_wallner](https://medium.com/@emilwallner/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d) original model we eventually achieved something:
- 1000 epochs with only one picture to train and test
- Train photo, test photo, result


| Train image             |  Test image     | Result image |
:-------------------------:|:-------------------------:|:------------------:
![](pics/res_0.jpg) |![](pics/test_0.jpg) |![](pics/img_0.png)

_versions necessary to make the model run in one's computer use tensorflow 1.14.0 and keras 2.1.6_
*(!pip install tensorflow==1.14.0)* *(!pip install keras==2.1.6)* *(h5py=2.10.0)*

### Implementing Siren


Firstly, quite difficult implementations were tried but at some point it occurred that the implementation can be done quite easily:

Here is the function to initialize the first layer of the model as described in the article altough the $\omega_0=1/30$ which is different than in the article:
(In the article $\omega_0=30$ but this provided poor results. We did not figured out why but $\omega_0=1/30$ in contrary worked fine!)

```
from keras.utils.generic_utils import get_custom_objects

def siren_in(x):
    x *= 1/30
    return tf.math.sin(x)

get_custom_objects().update({'siren_in': Activation(siren_in)})
```

and here is how it looks like in the layer:

`encoder_output = Conv2D(64, (3,3), activation="siren_in", kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1), padding='same', strides=2)(encoder_input)`

where `kernel_initializer` is for the weight initialization.

For other layers, the _siren_ could be given in such a way:

`encoder_output = Conv2D(128, (3,3), activation=tf.math.sin, kernel_initializer="he_uniform", padding='same')(encoder_output)`

where `he_uniform` is a uniform distribution $U(-\sqrt{\frac{6}{fan_{in}}},\sqrt{\frac{6}{fan_{in}}})$ (fan_in is the number of input units in the weight tensor) which is same as described in the paper.

### EMIL_WALLNER model testing

Many different combinations of activation functions were tried out but eventually three were chosen to carry out final experiments:
Let's call the models followingly:
- FULL_RELU(FR): everything is left as in the original model- only the ultimate layer's activation function is *tanh*, others are *relu*
- SIREN_RELU_SIREN(SRS): only the first and last layer are changed to have *siren* activation function
- SIREN130(S130) aka FULL_SIREN: all the layers are using *siren*

All the mentioned models proved rather satisfing results on fitting one picture. The results of fitting are represented below:

| Full_Relu(FR) |  Siren_Relu_Siren(SRS) | Siren130(S130)|
|:-------------------------:|:-------------------------:|:------------------:|
![](pics/gifs/relufit.gif) |![](pics/gifs/srsfit.gif)  |![](pics/gifs/siren130fit.gif)
|                          |*Final result after 1000 epochs* |               |
![](pics/gifs/relu_fit_only.png) |![](pics/gifs/srs_fit_only.png) |![](pics/gifs/siren130_fit_only.png)
|                       |   *Loss from 50 to 1000 epochs*     |              |
![](pics/gifs/reluloss.PNG) |![](pics/gifs/srs_loss.PNG) |![](pics/gifs/siren_loss.PNG)

We can see that the loss is less than ~0.002 respectively after 1000, 300 and 900 epochs. This is the first implication that _siren_ combined with _relu_ is converging remarkably faster then the others.

### Models trained on 40 images

As, also stated in the [emil_wallner](https://medium.com/@emilwallner/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d), first "proper" results started to emerge after training with 40 images. We chose 40 images, mainly portraits and some landscapes, and trained the model on them. For this, as it takes a lot of machine power already, we used Google Colab and adjusted the model to work with GPU.

We used 2 bach training with 20 images each and here is the loss after 1000 epochs of training:

| Full_Relu |  Siren_Relu_Siren | Siren130|
|:-------------------------:|:-------------------------:|:------------------:|
|                       |   *Loss from 50 to 1000 epochs*     |              |
![](pics/tests/relu/loss_relu.PNG) |![](pics/tests/srs/loss_srs.PNG) |![](pics/tests/siren/loss_siren.PNG)

We can see more or less the same results as before. They achieve loss ~0.002 respectively after 800, 300 and 1000 epochs but the validation loss at these points is approximately 0.0082, 0.009 and 0.009. So, looking at validation loss FR is still the best model but we cannot say that for sure as SRS is converging just so much faster.

Now, lets look at some visual testing and results, so that everyone could judge themselves which models predicts the best after this still rather small scale trainig action.

Predicted training images compared to ground truth

| Full_Relu |  Siren_Relu_Siren | Siren130 | Ground Truth |
|:-------------------------:|:-------------------------:|:------------------:|:----------------------------:|
|                          |*Predicting training images after*  |    *1000 epochs*         |           |
![](pics/tests/relu/relu_trn%20(5).png) |![](pics/tests/srs/srs_trn%20(1).png) |![](pics/tests/siren/siren_trn%20(5).png) |![](pics/tests/train/gt%20(2).jpg)
![](pics/tests/relu/relu_trn%20(2).png) |![](pics/tests/srs/srs_trn%20(3).png) |![](pics/tests/siren/siren_trn%20(2).png) |![](pics/tests/train/gt%20(3).jpg)
![](pics/tests/relu/relu_trn%20(3).png) |![](pics/tests/srs/srs_trn%20(4).png) |![](pics/tests/siren/siren_trn%20(3).png) |![](pics/tests/train/gt%20(1).jpg)

We can see that as they have all reached a different level of fitness. The SRS model has clearly overfitted due to its fast convergence. S130 has overfitted least.

Predicted validataion images compared to ground truth:

| Full_Relu |  Siren_Relu_Siren | Siren130 | Ground Truth |
|:-------------------------:|:-------------------------:|:------------------:|:----------------------------:|
|                          |*Predicting validation images*  |    *after 1000 epochs*         |           |
![](pics/tests/relu/relu_vld%20(2).png) |![](pics/tests/srs/srs_vld%20(1).png) |![](pics/tests/siren/siren_vld%20(1).png) |![](pics/tests/valid/org%20(1).jpg)
![](pics/tests/relu/relu_vld%20(3).png) |![](pics/tests/srs/srs_vld%20(2).png) |![](pics/tests/siren/siren_vld%20(2).png) |![](pics/tests/valid/org%20(2).jpg)
![](pics/tests/relu/relu_vld%20(4).png) |![](pics/tests/srs/srs_vld%20(3).png) |![](pics/tests/siren/siren_vld%20(3).png) |![](pics/tests/valid/org%20(3).jpg)
![](pics/tests/relu/relu_vld%20(5).png) |![](pics/tests/srs/srs_vld%20(4).png) |![](pics/tests/siren/siren_vld%20(4).png) |![](pics/tests/valid/org%20(4).jpg)
![](pics/tests/relu/relu_vld%20(1).png) |![](pics/tests/srs/srs_vld%20(5).png) |![](pics/tests/siren/siren_vld%20(5).png) |![](pics/tests/valid/org%20(5).jpg)

Predictions of the models for test data, where we do not know the ground truth:

| Full_Relu |  Siren_Relu_Siren | Siren130|
|:-------------------------:|:-------------------------:|:------------------:|
|                       | *Predicting test images after* |      *1000 epochs*         |
![](pics/tests/relu/exp11_relu.gif) |![](pics/tests/srs/exp11_srs.gif) |![](pics/tests/siren/exp11_siren.gif)
![](pics/tests/relu/exp12_relu.gif) |![](pics/tests/srs/exp12_srs.gif) |![](pics/tests/siren/exp12_siren.gif)
![](pics/tests/relu/exp13_relu.gif) |![](pics/tests/srs/exp13_srs.gif) |![](pics/tests/siren/exp13_siren.gif)
![](pics/tests/relu/exmp3_relu.gif) |![](pics/tests/srs/exmp3_srs.gif) |![](pics/tests/siren/exmp3_siren.gif)
![](pics/tests/relu/exmp2_relu.gif) |![](pics/tests/srs/exmp2_srs.gif) |![](pics/tests/siren/exmp2_siren.gif)
![](pics/tests/relu/ima_other2.png) |![](pics/tests/srs/ima_other5.png) |![](pics/tests/siren/ima_nut2.png)

We can see again that, as these results are saved after 1000 epochs and due that time SRS has overfitted a lot whose signes can be noticed easily from the results.

#### Conclusion

It is hard to say which of those models is really the best but on thing is sure, _siren_ activation function smartly combined with _relu_ converges approximately 3 times faster than other given models.

##  Fully connected linear layers SIREN colorization model

In this part, we use the model present in the paper:

```
img_siren = Siren(in_features=48*48, out_features=3*48*48, hidden_features=3*48*48, hidden_layers=3, outermost_linear=True)
```

It ensures that the model is correctly initialized and can learn from training.

The model trained consists of 3 layers fully connected with sinus activation function. In order to maintain a size of the model reasonable (less than 5Gb), the images are reduced to 48*48 pixels. The input consists of a black and white image, and the output is the RGB image. The model is trained on batches of size 70, on a dataset of 720 images representing beaches. In the following results, each image is provided from left to right as:
- output of the model
- initial RGB image
- input of the model: a black and white image

Each layer is fully connected: this model does not use convolution layers to identify categories of objects and shape in the image, but considers the entire image at once. This means that the model is larger in size, but also that the model has a direct perception of the entire image, hence knowing what the image represents.


Loss during training:

![loss](imgs/colorization/loss.png)

Training results:

![training_images](imgs/colorization/training_images_2.png)

Testing results:

![testing_images](imgs/colorization/testing_images_2.png)

We notice that the model overfits on the training dataset: it manages to obtain far better results on the training data than on the testing data. However, this results in the best outcome: the testing data manages to capture most of the shapes, and a partial colorization.

A similar model was also trained on a larger dataset: 7000 images of landscapes (beaches, mountains, forests, icecaps): the testing results were quite bad. This may come from the increased diversity of the images. Therefore, the model did not know which colors to apply depending on the type of images.

Another problem is the size of the model: creating fully connected layers results in a model size proportionate to the square of the resolution of the image. Therefore, this model can only be applied on small images: in order to obtain a colorization of a large image, it would have to be split in small pieces, and then reassembled.


As a conclusion, SIREN models are capable of associating correct colors on images. However, the training can only be successful on classes of images: thee colorization must be able find similar images in the training dataset in order to be successful on other images. In addition, the colorization should be combined on small parts of images in order to obtain a quality colorization.


# Merging

## Steps of transformation

We consider image A and image B, we want to obtain image C as a mix of the 2 initial images. If we merge the images pixel by pixel, we will obtain new colors which we do not want. instead, we want only the main elements of each image to be present. Therefore, we are going to merge the gradients of the 2 images, and then build the image associated to this gradient. This operation does not require the use of SIREN networks. However, it is possible to use them as a storage format of each image.

![Model](imgs/SIREN_merging.png)

The model used is taken from the article.

### First test: merging 2 black and white images

We take 2 images 128*128 in black and white and merge them together. The results for each image are:
- image outputted by SIREN model
- gradient off the output
- laplacian of the output

The images we show are image A, B, and C.

![image_A_bw](imgs/image_A_bw.png)
![image_B_bw](imgs/image_B_bw.png)
![image_C_bw](imgs/image_C_bw.png)


We see that the resulting image has the details of both images. Since the elements are in the same positions in each image, it results is transparent shapes.

### Second test: merging 2 RGB images

This step required creating the gradients of the functions by hand: for computational efficiency, we only computed the horizontal gradient of each image. We obtained the merged image as follows:

![image_A_rgb](imgs/image_A_rgb.png)
![image_B_rgb](imgs/image_B_rgb.png)
![image_C_rgb](imgs/image_C_rgb.png)

Although the gradient is only partially computed, the 2 images are correctly merged, with colors being respected. We notice that the precision of the image is reduced. However, this is likely to come only from the gradient formula used. This result is very favorable: SIREN allows merging images so that only main elements of each initial image appear on the final merged image. An application of this is the creation of photoshopped images. Although it is possible not to use SIREN to merge images (simply computing the mean of the gradients), SIREN provides a structure on sets of images: SIREN stored images form a group on which they can be added to obtain realistic additional images.



# Resizing images using SIREN

...
