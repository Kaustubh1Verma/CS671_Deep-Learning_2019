import numpy as np
import tensorflow.keras as keras
import argparse as arg
from pathlib import Path

def ensure_folder(folder):
	p = Path(folder)
	if not p.is_dir():
		print(p, 'doesn\'t exist!')
		print('creating', p)
		p.mkdir(parents=True)
	return True		

def ensure_file(file):
	p = Path(file)
	if not p.is_file():
		print(p, 'doesn\'t exist!')
		raise SystemExit
	return True		

def main():
	parser = arg.ArgumentParser()
	parser.add_argument('model', help='/path/to/the/model.h5')
	parser.add_argument('test_data', help='/path/to/input/data.npz array named \'test\' must be in shape (None, 400, 250, 1)')
	parser.add_argument('output_dest_dir', help='/path/to/output/folder output will be saved here. default:\'.\'', nargs='?', default='.')
	args = parser.parse_args()

	ensure_file(args.model)
	print('model to load:', args.model)
	ensure_file(args.test_data)
	print('test data to load:', args.test_data)
	ensure_folder(args.output_dest_dir)
	print('output will be stored as', Path(args.output_dest_dir)/'output.npy')
	print('loading model', args.model)
	model = keras.models.load_model(args.model)
	print('loading test data', args.test_data)
	test = np.load(args.test_data)['test']
	print('Predicting Core Points')
	output = model.predict(test)
	print('Saving output to', Path(args.output_dest_dir)/'output.txt')
	np.savetxt(Path(args.output_dest_dir)/'output.txt', output, fmt='%d')
	print('-------- Done -------')
	print('Group 04 - Assignment 03 - Task 03')

if __name__ == '__main__':
	main()