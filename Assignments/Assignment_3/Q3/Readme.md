# Files

## corepoint.py
Runs the `model.h5` on any test data. Also saves the output in a txt file.

## model.h5
The trained model that can find the core point of a fingerprint.

## preprocess.py
Scales all images in a folder to 250x400. The corresponding ground truth will also be mapped onto the new size of the image. For this the folder with ground truths should also be provided. Next, it will save the test data into a compressed npz file. All the ground truths will also be saved in a txt for easy comparison later on. 

## q3.ipynb
The notebook where the model was trained.