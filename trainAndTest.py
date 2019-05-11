import loadAndSampleData
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical

# Here we sill split up the data for training/validation/testing
def split(X, y, split_perc = 0.15):
    print("splitting at", str(split_perc) + "%")
    X_train, X_test, y_train, y_test, = train_test_split(X,y,test_size=split_perc, random_state = 42)
    print("X_train",X_train.shape)
    print("X_test",X_test.shape)
    print("y_train",y_train.shape)
    print("y_test",y_test.shape)
    return X_train, X_test, y_train, y_test

def run(X_train, X_test, y_train, y_test, num_cat = 10, dropout_rate = 0.2):
    model = Sequential()

    # Caculate the mean across all training examples
    mu = np.mean(X_train)
    # Caculate the standard deviation across all training examples
    sigma = np.std(X_train) + 10
    #Normalize pixels
    # The input shape might have to be size 28x28x1
    X_train = X_train.reshape((X_train.shape[0],28,28,1))
    X_test = X_test.reshape((X_test.shape[0],28,28,1))

    #Convert our labels to one-hot vecotrs
    y_train = to_categorical(y_train, num_cat, dtype='float32')
    model.add(Lambda(lambda x: x - mu/sigma, input_shape = (28,28,1)))
    model.add(Conv2D(5, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(5, (3,3), padding = "same", activation= "relu"))
    model.add(Conv2D(5, (3,3), padding = "same", activation= "relu"))
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(Dense(700, activation = "relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(500, activation = "relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(400, activation = "relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(300, activation = "relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(200, activation = "relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(100, activation = "relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(50, activation = "relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation = "softmax"))
    # Use the adam optimizer
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    print()
    print("Fitting the model")

    #Start training
    #TODO: Change shuffle to False and change nb_epoch to 12 or 32
    history_object = model.fit(X_train, y_train, validation_split = 0.15, shuffle = False, nb_epoch=32)
    print("model fitted")
    print(history_object.history.keys())

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model cross entropy loss')
    plt.ylabel('cross-entropy loss')
    plt.xlabel('epoch')
    plt.legend(['training_set', 'validation_set'], loc='upper right')
    plt.show()
    return

def main():
    X, y, categories = loadAndSampleData.main(True)
    print("begin training")
    X_train, X_test, y_train, y_test = split(X,y)
    run(X_train, X_test, y_train, y_test, len(categories))
    return

if __name__ == "__main__":
	main()
