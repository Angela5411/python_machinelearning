# Import Packages
import numpy as np
from sklearn.model_selection import train_test_split
from mnist import MNIST
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, UpSampling2D, Activation
batch_size = 128
num_classes = 10
epochs = 40

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
mndata = MNIST('C:\Program Files\Python37\Lib\site-packages\mnist\samples')
X_test, y_test = mndata.load_testing()
X_test,y_test = np.array(X_test), np.array(y_test)
X_train, y_train = mndata.load_training()
X_train,y_train = np.array(X_train),np.array(y_train)
print(np.shape(X_train))

# do not forget the validate set
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

# reshaping the data to appropriate tensor format
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_validate = X_validate.reshape(X_validate.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print(np.shape(X_train))

# Build the autoencoder
model = Sequential()
model.add(Conv2D(14, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(14, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(1, kernel_size=3, padding='same', activation='relu'))
model.compile(optimizer='adam', loss="mse")
model.summary()

# Train the model
model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_validate, X_validate), verbose=1)
# saving the trained model
model_name = './OutputData/Autoencoder.h5'
model.save(model_name)

print('CNN topology setup completed')

#