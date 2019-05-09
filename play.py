import numpy as np
from matplotlib import pyplot as plt

print('start')
filename='./data/full_numpy_bitmap_The Eiffel Tower.npy'
print(filename)
images = np.load(filename)
print('type of images: ',type(images))
print('shape of images: ',images.shape)
image=images[0]
print('type of image: ',type(image))
print('shape of image: ',image.shape)
reshapedImage=image.reshape((28,28))
print('after reshape.')
print('type of image2: ',type(reshapedImage))
print('shape of image2: ',reshapedImage.shape)
# for i in range(len(reshapedImage)):
#     for j in range(len(reshapedImage[i])):
#         print(reshapedImage[i][j], end=" ")
#     print()
plt.imshow(reshapedImage, cmap='gray', interpolation='nearest');
plt.show()
#import sys
#data = sys.stdin.readlines()
