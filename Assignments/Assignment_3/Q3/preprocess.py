import numpy as np
from multiprocessing import Pool
from pathlib import Path
from PIL import Image
import argparse

def ensure_folder(folder):
	p = Path(folder)
	if not p.is_dir():
		print(p, 'doesn\'t exist!')
		print('creating', p)
		p.mkdir(parents=True)
	return True		

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def imresize(imagepath, w, h, oldgt_folder, img_dest_folder, gt_dest_folder):
	newsize = np.array([w, h], dtype=float)
	img = Image.open(imagepath)
	img = img.convert('L')
	oldsize = np.array(img.size)
	newimg = img.resize((w, h), resample=Image.BICUBIC)
	img.close()
	if not img_dest_folder.is_dir():
		img_dest_folder.mkdir(parents=True, exist_ok=True)
	newimg.save(img_dest_folder/(imagepath.name))

	oldgt = np.loadtxt(oldgt_folder/(imagepath.name[:-5]+"_gt.txt"), dtype=float)
	oldgt = np.flip(oldgt)
	newgt = (oldgt / oldsize) * newsize
	if not gt_dest_folder.is_dir():
		gt_dest_folder.mkdir(parents=True, exist_ok=True)
	np.savetxt(gt_dest_folder/(imagepath.name[:-5]+"_gt.txt"), newgt, fmt='%d')
	del newimg
	return 1

def main():
	print('----------- Group 04 - Assignment 03 - Task 03 - Preprocessing ---------------------')
	parser = argparse.ArgumentParser()
	parser.add_argument('images_folder', help='/path/to/dir/with/images/')
	parser.add_argument('ground_truth_folder', help='/path/to/dir/with/ground/truth')
	parser.add_argument('images_output_folder', help='/output/folder/of/new/images default=\'newimages\'', nargs='?', default='newimages')
	parser.add_argument('ground_truth_output_folder', help='/output/folder/of/new/gt default=\'newgt\'', nargs='?', default='newgt')
	args = parser.parse_args()
	if not Path(args.images_folder).is_dir():
		print(args.images_folder, 'doesn\'t exist.')
		raise SystemExit
	print('images will load from', args.images_folder)
	if not Path(args.ground_truth_folder).is_dir():
		print(args.ground_truth_folder, 'doesn\'t exist.')
		raise SystemExit
	print('Ground truth will load from', args.ground_truth_folder)
	ensure_folder(args.images_output_folder)
	print('new images will be saved at', args.images_output_folder)
	ensure_folder(args.ground_truth_output_folder)
	print('new gt will be saved at', args.ground_truth_output_folder)

	p = Pool(4)
	arguments = [(img, 250, 400, Path(args.ground_truth_folder), Path(args.images_output_folder), Path(args.ground_truth_output_folder)) for img in Path(args.images_folder).iterdir()]
	print('transforming images...')
	res = p.starmap(imresize, arguments)
	
	print('loading all images from', args.images_output_folder)
	x = np.array([np.array(Image.open(img)) for img in Path(args.images_output_folder).iterdir()])[..., np.newaxis]
	print(x.shape)
	print('saving npz at', 'test.npz')
	np.savez_compressed('test.npz', test=(x/255))
	del x

	print('loading all gt from', args.ground_truth_output_folder)
	gt = np.array([np.loadtxt(f) for f in Path(args.ground_truth_output_folder).iterdir()])
	print('saving gt at', 'gt.txt')
	np.savetxt('gt.txt', gt, fmt='%d')
	del gt
	
if __name__ == '__main__':
	main()