import tensorflow as tf

def dense(x, in_length, neurons, activation, layer_name, 
	wdev=1.0, bdev=1.0, dev=None):
	'''dense(x, in_length, neurons, activation, layer_name)
	Parameters
	----------
	x: Array
	in_length: int, length of array
	neurons: int, number of neorons in the layer
	activation: A callable activation function
	layer_name: string, A name for this layer. This is used for the
		variable scope
	dev: float, the standard deviation of gaussian distribution used to 
		generate random values. Tweaking this can give significantly 
		different results. This is applied only to weights. 

	Returns
	-------
	activation(w*x + b) 
	'''
	if(dev!=None):
		wdev = dev
		bdev = dev
	with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
		w = tf.get_variable("w", [in_length, neurons], 
			initializer=tf.initializers.random_normal(stddev=wdev))
		b = tf.get_variable("b", [neurons], 
			initializer=tf.initializers.random_normal(stddev=bdev))
	return activation(tf.add(tf.matmul(x, w), b))
