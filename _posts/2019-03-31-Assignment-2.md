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
