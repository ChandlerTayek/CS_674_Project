import numpy as np
from matplotlib import pyplot as plt
import random
import sys

def processFile(filename, cat, sup_output = False, verbose = False):
	if verbose:
		print("filename",filename)
	images = np.load(filename)
	number_of_exp = images.shape[0]
	if not sup_output:
		print('Number of',cat,'images: ', number_of_exp)
	return images
def reshapeImages(list_images, verbose = False):
	reshaped_list = []
	for cat in list_images:
		if verbose:
			print(cat.shape)
		reshaped_img = cat.reshape((cat.shape[0],28,28))
		reshaped_list.append(reshaped_img)
		if verbose:
			print(reshaped_img.shape)
	return reshaped_list
def loadUpData(cat, sup_output = False):
	list_of_images_by_cat = []
	for category in cat:
		filename='./data/full_numpy_bitmap_'+category+'.npy'
		list_of_images_by_cat.append(processFile(filename, category, sup_output))
	return list_of_images_by_cat


def random_sample(list_cat_imgs, num_samples, cat_english_labels, sup_output = False, verbose = False):
	if sup_output:
		verbose = False

	#First we need to check that the number of samples is smaller than or equal
	#equal to the smallest number of examples for any category
	min_number_examples = list_cat_imgs[0].shape[0]
	for cat in list_cat_imgs:
		examples = cat.shape[0]
		if examples < min_number_examples:
			min_number_examples = examples
	if num_samples > min_number_examples:
		if not sup_output:
			print("too many samples and not enough examples")
		return list_cat_imgs

	resampled_cat_imgs = []

	#TODO: If ou have time change the list structure to a dict so you don't
	# 	   to do this nasty index var 'i' below to associate the label with
	#      the data
	i = 0
	for cat in list_cat_imgs:
		number_of_training_examples = cat.shape[0]
		if not sup_output:
			print('Take',num_samples, 'samples from', number_of_training_examples, 'of', cat_english_labels[i] + 's')

		#Uniformly select num_samples worth of samples from X
		#uncomment seed if you want it to not produce the same results
		random.seed(1)
		idx = random.sample(range(number_of_training_examples),num_samples)
		# Select only the samples that were randomly generated
		samplesX = cat[idx,:]
		resampled_cat_imgs.append(samplesX)
		i += 1
		if verbose:
			print("index from samples")
			print(idx)
			print("samples drawn")
			print(samplesX)
			print(samplesX.shape)
	return resampled_cat_imgs

def squish(uniform_list):
	#For this to work the list needs to have all of the same num of examples
	number_of_examples_per_Cat = uniform_list[0].shape[0]
	number_of_cat = len(uniform_list)
	total_examples = number_of_examples_per_Cat*number_of_cat
	X = np.zeros(shape = (total_examples,28,28), dtype = int)
	i = 0
	for cat in uniform_list:
		lidx = i*number_of_examples_per_Cat
		ridx = (i+1)*number_of_examples_per_Cat
		X[lidx:ridx,:,:] = cat
		i += 1
	return X



def expand_labels(num_cat, samples):
	y = np.zeros((num_cat*samples),dtype=int)
	for i in range(num_cat):
		lidx = i*samples
		ridx = (i+1)*samples
		y[lidx:ridx] = np.repeat(i, samples)
	return y

def print_shapes_list(lis):
	for cat in lis:
		print(cat.shape)
	return

def main(sup_outs = False):
	categories = ['cannon','eye', 'face', 'nail', 'pear','piano','radio','spider','star','sword']
	samples = 10000
	num_cat = len(categories)
	list_of_cat_images = loadUpData(categories, sup_output = True)
	sub_sampled_imgs = random_sample(list_of_cat_images, num_samples = samples, cat_english_labels = categories,sup_output=False)
	resahped_list_imgs = reshapeImages(sub_sampled_imgs)
	# print_shapes_list(resahped_list_imgs)
	X = squish(resahped_list_imgs)
	y = expand_labels(num_cat, samples)
	return X, y, categories

if __name__ == "__main__":
	main()
