# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from mnist import MNIST
from keras.layers import Dropout, Conv2D, MaxPool2D, UpSampling2D, Activation, Flatten
from keras import backend as K
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import seaborn as sns

batch_size = 128
num_classes = 10
epochs = 2

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# mndata = MNIST('C:\Program Files\Python37\Lib\site-packages\mnist\samples')
mndata = MNIST('C:\Program Files\Python37\Lib\site-packages\mnist\samples')
X_test, y_test = mndata.load_testing()
X_test,y_test= array(X_test), array(y_test)
X_train, y_train = mndata.load_training()
X_train,y_train=array(X_train),array(y_train)

# X_test = X_test.reshape(len(X_test), 28, 28)
# X_train = X_train.reshape(len(X_train), 28, 28)
# # plot 9 images as gray scale
# plt.subplot(331)
# plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
# plt.subplot(332)
# plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
# plt.subplot(333)
# plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
# plt.subplot(334)
# plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# plt.subplot(335)
# plt.imshow(X_train[4], cmap=plt.get_cmap('gray'))
# plt.subplot(336)
# plt.imshow(X_train[5], cmap=plt.get_cmap('gray'))
# plt.subplot(337)
# plt.imshow(X_train[6], cmap=plt.get_cmap('gray'))
# plt.subplot(338)
# plt.imshow(X_train[7], cmap=plt.get_cmap('gray'))
# plt.subplot(339)
# plt.imshow(X_train[8], cmap=plt.get_cmap('gray'))
# plt.show()
print(np.shape(X_train))
# do not forget the validate set
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.1, random_state=1)
print(np.shape(X_train))

# # reshaping the data to appropriate tensor format
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_validate = X_validate.reshape(X_validate.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print(np.shape(X_train))

# X_train = np.expand_dims(X_train, axis=1)
# X_validate = np.expand_dims(X_validate, axis=1)
# X_test = np.expand_dims(X_test, axis=1)
# print(np.shape(X_train))



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
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_validate, y_validate), verbose=1)

# Fitting testing dataset
restored_testing_dataset = model.predict(X_test)

# Observe the reconstructed image quality
plt.figure(figsize=(20,5))
for i in range(10):
    index = y_test.tolist().index(i)
    plt.subplot(2, 10, i+1)
    plt.imshow(X_test[index].reshape((28,28)))
    plt.gray()
    plt.subplot(2, 10, i+11)
    plt.imshow(restored_testing_dataset[index].reshape((28,28)))
    plt.gray()

# Extract the encoder
encoder = K.function([model.layers[0].input], [model.layers[4].output])

# Encode the training set
encoded_images = encoder([X_test])[0].reshape(-1, 7*7*7)

# Cluster the training set
kmeans = KMeans(n_clusters=10)
clustered_training_set = kmeans.fit_predict(encoded_images)

# Observe and compare clustering result with actual label using confusion matrix
cm = confusion_matrix(y_test, clustered_training_set)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()

# Plot the actual pictures grouped by clustering
fig = plt.figure(figsize=(20,20))
for r in range(10):
    cluster = cm[r].argmax()
    for c, val in enumerate(X_test[clustered_training_set == cluster][0:10]):
        fig.add_subplot(10, 10, 10*r+c+1)
        plt.imshow(val.reshape((28,28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('cluster: '+str(cluster))
        plt.ylabel('digit: '+str(r))

