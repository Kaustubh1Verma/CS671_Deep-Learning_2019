---
layout	: post
title	: "Assignment 1"
comments	: false
author	: "Group 04"
---

All code for this assignment can be found [here](https://github.com/42niks/CS671-Deep-Learning-2019/tree/master/Assignments/Assignment_1).

# Question 1 - Aj R Laddha
# Question 2 - Kaustubh Verma
# Question 3 - Nikhil T R
## Description
In order to fully understand a fully connected layer, in this problem we are asked to create a dense layer api on top of lower level tensorflow functions like multiply. From the very beginning I've been complaining about having to work on tf1 when tf2 is being released, especially when tf1 api is being ridiculed by many.<br>However, I am ambivalent. I also believe that doing this will help me appreciate deep learning on a _deeper_ level (lol). The underlying concepts will not change, and this should be my only focus.<br>
Using this api, I am asked to make a model to classify a given MNIST data set and the dataset that we've made in [Question 1](#question-1---aj-r-laddha).

## My work
The api that I've designed basically uses everything from tensorflow except the layer api. My layer api can be found here [here](https://github.com/42niks/CS671-Deep-Learning-2019/blob/master/Assignments/Assignment_1/Q3/layers.py). I implemented only dense as that was the only requirement. The dense function creates a new `tf.variable_scope` for a dense layer. This gives me the added advantage of naming weights as `w` and bias as `b` everytime. It also accepts values for the standard deviation of the normal distribution from which the `Tensors` are initialized. For usage, refer to the notebooks [here](https://github.com/42niks/CS671-Deep-Learning-2019/tree/master/Assignments/Assignment_1/Q3)