import loadAndSampleData
from matplotlib import pyplot as plt
import numpy as np



#+ cat_lab[label]))
def visualizeRandomSample(X,y,cat_lab):
    index = np.random.randint(0, len(X))
    label = y[index]
    image = X[index,:,:].squeeze()
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.title(label = (cat_lab[label]))
    plt.show()
    return



def main():
    X, y, categories = loadAndSampleData.main("arg passed")
    visualizeRandomSample(X,y,categories)
    return

if __name__ == "__main__":
	main()
