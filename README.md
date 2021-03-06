# Naive Background Style Transfer

Naive Background Style Transfer implemented using Keras and TensorFlow by Vikram Shenoy.

## Overview

Naive background style transfer performs style transfer only on the background of the given content image. This naive approach uses two networks to reproduce this style transfer. The DeepLabv3+ model generates a segmentation map of the image which is processed to create a binary mask highlighting the foreground from the background. The style transfer model uses this segmented mask to guide the stylized pixels only onto the background of the given content image. For more details, check out the [*report*](https://github.com/VikramShenoy97/Naive-Background-Style-Transfer/blob/master/Report/Naive_Background_Style_Transfer_Final.pdf).

![Architecture](https://github.com/VikramShenoy97/Naive-Background-Style-Transfer/blob/master/Media/Architecture.png)

The above architecture shows how the algorithm works and creates the resulting output image.

The picture below is of me taken on my trip to Amsterdam.

![Content](https://github.com/VikramShenoy97/Naive-Background-Style-Transfer/blob/master/Media/shen.jpg)


The picture displayed here is the result of Naive Background Style Transfer in the style of one of my favorite pieces of art, *The Starry Night* by Vincent Van Gogh.


![Style_Transfer](https://github.com/VikramShenoy97/Naive-Background-Style-Transfer/blob/master/Media/Style_Transfer.jpg)


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

For using Naive Background Style Transfer, you need to install Keras, TensorFlow, PIL, and Imageio.

```
pip install keras
pip install tensorflow
pip install Pillow
pip install imageio
```

### Images


**Content Image**       |  **Style Image**
:-------------------------:|:-------------------------:
![Original](https://github.com/VikramShenoy97/Naive-Background-Style-Transfer/blob/master/Input_Images/Content/portrait.jpg)  |  ![Style](https://github.com/VikramShenoy97/Naive-Background-Style-Transfer/blob/master/Input_Images/Style/scream.jpg)

### Run

Run the script *main.py* in the terminal as follows.

```
Python main.py
```

## Results
The final output is stored in Output Images.

### Intermediate Stages of Style Transfer

Here is the generated image through different intervals of the run.

![Intermediate_Image](https://github.com/VikramShenoy97/Naive-Background-Style-Transfer/blob/master/Output/Final_Image/Intermediate_Images.jpg)

### Transition through epochs

![Transition](https://github.com/VikramShenoy97/Naive-Background-Style-Transfer/blob/master/Output/Animation/nbst_animation.gif)

### Result of Style Transfer

![Final_Image](https://github.com/VikramShenoy97/Naive-Background-Style-Transfer/blob/master/Output/Final_Image/Style_Transfer.jpg)


## Ablation Study of Hyperparameters
This section takes a closer look at the hyperparameters of NBST, mainly the content weight and the style weight.

### Default Parameters
• Content Weight = 1e3


• Style Weight = 1e-2


![default](https://github.com/VikramShenoy97/Naive-Background-Style-Transfer/blob/master/Media/Style_Transfer.jpg)


### High Content Weight
• Content Weight = 1e10


• Style Weight = 1e-2


![content_increase](https://github.com/VikramShenoy97/Naive-Background-Style-Transfer/blob/master/Media/cw1e10.jpg)


### High Style Weight
• Content Weight = 1e3


• Style Weight = 1e15


![style_increase](https://github.com/VikramShenoy97/Naive-Background-Style-Transfer/blob/master/Media/sw1e15.jpg)


## Built With

* [Keras](https://keras.io) - Deep Learning Framework
* [TensorFlow](https://www.tensorflow.org) - Deep Learning Framework
* [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb) - Cloud Service

## Authors

* **Vikram Shenoy** - *Initial work* - [Vikram Shenoy](https://github.com/VikramShenoy97)

## Acknowledgments

* **Leon A. Gatys's** paper, [*A Neural Algorithm of Artistic Style*](https://arxiv.org/abs/1508.06576)
* **Liang-Chieh Chen's** paper, [*Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*](https://arxiv.org/pdf/1802.02611.pdf)
* Project makes use of the [*Pre-trained DeepLabv3+ Model*](https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb#scrollTo=aUbVoHScTJYe) by Google
* Content Image: Scraped from Google Image Search, [*Aila Images*](https://www.shutterstock.com/video/search?contributor=Aila+Images)
* Style Image: The Scream by Edvard Munch, [*The Scream*](https://en.wikipedia.org/wiki/The_Scream)
* Style Image: Starry Night by Vincent Van Gogh, [*The Starry Night*](https://en.wikipedia.org/wiki/The_Starry_Night)
