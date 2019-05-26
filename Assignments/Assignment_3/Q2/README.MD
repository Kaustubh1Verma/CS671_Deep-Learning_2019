
### Pixelwise Image Segmentation

TEST CODE ON TEST IMAGES:(Using script.py)

1. Put all the test images in a folder named such as "test_images". Ensure that this folder,and model_weights are in the same path as script.py.
2. Run the code using the command - 
		python script.py -T test_images -O predicted_mask
	where predicted_mask is the name of the output folder. Do not put "/" after name of "test_images" or "predicted_mask". The "predicted_mask"
	directory will be created if not already present.
3. The mask will be made in predicted mask folder.



####################################### Training ######################################

### Preparing resized and renamed images 
Using code "Prepare_train.ipynb", and bp putting original files in Q2/Data,Q2/Mask and train.pickle(which contains corresponding file name) prepare "resized_data". 

Now the data in resized_data/data and resized_data/mask can be used to train the model.
The weights get saved in model_weights_full.


################## NOTE ###################

* Download the model weights from https://cloud.iitmandi.ac.in/d/ceb988b684/ and put it inside the model_weights_full folder.
* Also download VGG-weights(vgg16_weights_tf_dim_ordering_tf_kernels_notop) from https://cloud.iitmandi.ac.in/f/d90ca19789/?raw=1 and place it in the same folder as training.
