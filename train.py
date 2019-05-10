import loadAndSampleData
from sklearn.model_selection import train_test_split

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


def main():
    X, y, categories = loadAndSampleData.main("arg passed")
    print("begin training")
    return

if __name__ == "__main__":
	main()
