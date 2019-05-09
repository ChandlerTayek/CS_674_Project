import numpy as np
from matplotlib import pyplot as plt

def processImage(image):
	#print('\ttype of image: ',type(image),', shape of image: ',image.shape)
	reshapedImage=image.reshape((28,28))
	# print('after reshape.')
	# print('type of image2: ',type(reshapedImage))
	# print('shape of image2: ',reshapedImage.shape)
	#for i in range(len(reshapedImage)):
	#	for j in range(len(reshapedImage[i])):
	#		print(reshapedImage[i][j], end=' ')
	#	print()
	#plt.imshow(reshapedImage, cmap='gray', interpolation='nearest');
	#plt.show()

def processFile(filename, cat):
	# print(filename)
	images = np.load(filename)
	print('Number of',cat,'images: ', images.shape[0])

	for image in images:
		processImage(image)

categories = ['cannon','eye', 'face', 'nail', 'pear','piano','radio','spider','star','sword']
for category in categories:
	filename='./data/full_numpy_bitmap_'+category+'.npy'
	processFile(filename, category)
