---
layout	: post
title	: "Assignment 3"
comments	: false
author	: "Group 04"
---

Full problem statements can be found [here](https://github.com/42niks/CS671-Deep-Learning-2019/blob/master/Assignments/Assignment_3/CS671_Assignment_3_2019_.pdf). All code for this assignment can be found [here](https://github.com/42niks/CS671-Deep-Learning-2019/tree/master/Assignments/Assignment_3).

# Question 1 - Aj R Laddha
# Question 2 - Kaustubh Verma
# Question 3 - Core Point Detection - Nikhil T R

## Description
In this task were asked to build and train a neural network that can find the core point of a given fingerprint. The dataset provided had 4000 images. Test images were not provided. 

## My Work
Initial look into the dataset showed 14 images that were polluted - they were blank. Those images were removed. I resized all the images to 250x400. The corresponding ground truths were also modified. A npz compressed archive of the images was made, along with the ground truths which was saved in a txt file for comparison later on. Scripts were also made to run the trained model on new test data.

### The Model - Explained
Since the final output are co-ordinates, this is basically a regression problem. The CNN block consisted of 3 layers with kernel size 3x3 and padding being `valid`. BatchNormalization was used for reducing overfitting and MaxPooling2D was used to bring down the feature size. The dense block was made of 3 layers with 128, 64 and 2 neurons respectively. 

Input images were normalized to the range of 0-1 from 0-255. However, the output was not normalized. The activation of the last layer was set to `relu`.