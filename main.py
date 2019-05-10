import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import random


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

# Here we sill split up the data for training/validation/testing
def split():
	X, y = np.arange(10).reshape((5,2)),range(5)
	print(X)
	print(list(y))
	X_train, X_test, y_train, y_test, = train_test_split(X,y,test_size=0.33, random_state = 42)
	print("X_train",X_train)
	print("X_test",X_test)
	print("y_train",y_train)
	print("y_test",y_test)
	print(train_test_split(y, shuffle=False))
	print(train_test_split(y))
	return

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
	number_of_examples_per_Cat = uniform_list[0].shape[0]
	number_of_cat = len(uniform_list)
	total_examples = number_of_examples_per_Cat*number_of_cat
	X = np.zeros(shape = (total_examples))
	for cat in uniform_list:

	retrun
def visualizeRandomSample():
	#plt.imshow(reshapedImage, cmap='gray', interpolation='nearest');
	#plt.show()
	return

def print_shapes_list(lis):
	[print(cat.shape) for cat in lis]
	return

def main():
	# categories = ['cannon','eye', 'face', 'nail', 'pear','piano','radio','spider','star','sword']
	# y = range(10)
	# list_of_cat_images = loadUpData(categories, True)
	# sub_sampled_imgs = random_sample(list_of_cat_images, num_samples = 1000, cat_english_labels = categories,sup_output=False)
	# resahped_list_imgs = reshapeImages(sub_sampled_imgs)

	# print_shapes_list(resahped_list_imgs)

	split()
	return

if __name__ == "__main__":
	main()
