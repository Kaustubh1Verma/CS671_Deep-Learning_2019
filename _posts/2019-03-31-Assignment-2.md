---
layout	: post
title	: "Assignment 2"
comments	: false
author	: "Group 04"
---

Full problem statements can be found [here](https://github.com/42niks/CS671-Deep-Learning-2019/blob/master/Assignments/Assignment_2/CS671_Assignment_2.pdf). All code for this assignment can be found [here](https://github.com/42niks/CS671-Deep-Learning-2019/tree/master/Assignments/Assignment_2).

# Question 1 - Foundations of Convolutional Neural Networks - Aj R Laddha
# Question 2 - Multihead Classification - Nikhil T R
The report and relevant notebooks can be found [here](https://github.com/42niks/CS671-Deep-Learning-2019/tree/master/Assignments/Assignment_2/Q2).

## Description
We were tasked with making a non sequential model using the Funcional API in keras. Basically, we had to classify the line dataset made in the [previous assignment](https://42niks.github.io/CS671-Deep-Learning-2019/2019/Assignment-1/), using multiple classification heads for each of the 4 categories: length, width, colour and angle. These 4 classifiaction heads were supposed to accept a feature map exatracted using a CNN based feature extractor. Length, width and colour classification heads were suppposed to use binary crossentropy as the loss function, whereas angle classification head was to use categorical crossentropy.

## My Work
I made the model using the [keras functional api](https://keras.io/getting-started/functional-api-guide/). The appropriate loss functions were set. The optimiser, adam in this case, acted upon the sum of all losses, which was the final loss.<br>
The major observation was the varying learning rates of the different classification heads. So, for the [third variation](https://github.com/42niks/CS671-Deep-Learning-2019/blob/master/Assignments/Assignment_2/Q2/Q2-v3.ipynb), I set loss weights during `model.compile`. The final loss is the weighted sum of all the 4 losses. This _sorta_ normalized the issue and I managed to get good performance on all 4 heads. Please refer to the [report](https://github.com/42niks/CS671-Deep-Learning-2019/blob/master/Assignments/Assignment_2/Q2/CS671_DL_A2.pdf) for more information.

# Question 3 - Kaustubh Verma

Report and Notebook for the code can be found [here](https://github.com/42niks/CS671-Deep-Learning-2019/tree/master/Assignments/Assignment_2/Q3)

## Description
This task involved visualizing of the outputs in a neural network. Such visualization especially in a CNN network helps to look at what sort of learning is happening and what portion is the network looking to produce the output. The three visualizations that we are performing here are -

**1. Visualizing Intermediate Layer Activations** - This visualization helps to take a look at the
different images from Convolution layers filters,and see how different filters in different layers
activate different parts of the image.
**2. Visualizing Convnet Filters** - In this visualizations ,we observe how different filters are learned
along the network by using Gradient Descent on value of convnet.
**3. Visualizing Heatmaps of class activations** - In this visulization we produce heatmaps of class
activations over input images.A class activation map is a basically a 2D grid of scores for a
particular output class for each location in the image.

## Procedure
1. For visualizing Intermediate Layer Activations it is the simple task of visualizing heatmap of activation at intermediate convolutional layers.
2. For visualizing Convnet Filters we can observe filters as images by running Gradient Descent on the value of
a convnet maximizing the response of a specific filter, starting from a blank input image.
3. For visualizing Heatmaps of class activations we implemented Grad-CAM(Gradient-weighted Class Activation Mapping) that uses class-specific gradient information flowing into the final convolutional layer of a CNN and produces a localization map of the important regions in the image.

