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