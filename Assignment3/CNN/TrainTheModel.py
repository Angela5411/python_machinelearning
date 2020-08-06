# TrainTheModel.

# Sfyridaki Angeliki cs151036
# Mhxanikwn plhroforikhs
# cs151036@uniwa.gr
# Part 1

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

batch_size = 128
num_classes = 10
epochs = 40
label_names = []
# input image dimensions
img_rows, img_cols = 32, 32


def classNames(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='ASCII')
    return dict['label_names']


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        # data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
        # The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
        # The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
        # labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
        X = np.array(dict[b'data'])
        Y = np.array(dict[b'labels'])
        x = []
        for i in X:  # 100000 rows/images
            x.append(np.reshape(np.transpose(np.reshape(i, (3, 1024))), (32, 32, 3)))
            # reshape into 3 rows(r,g,b) --> transpose so that each row refers to a pixel--> reshape so that each row is a row of 32 pixels
    return x, Y


# we read all the data that are necessary for our code
try:
    label_names = classNames('../InputData/dataset/batches.meta')
except:
    print('error while reading class names')
else:
    print('successfully read class names')

try:
    fileNameToLoad = '../InputData/dataset/data_batch_'  # define the txt file to pass the examples
    listX, listY = (unpickle(fileNameToLoad + str(1)))
except:
    print('error while reading data')
    exit(2)
else:
    print('successfully read data')

X = np.array(listX)
Y = np.array(listY)

# data are split between train and test sets
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, train_size=0.97, test_size=0.03)
print('data creation completed')
print('we have ', np.shape(x_train)[0], 'training paradigms and ', np.shape(x_valid)[0], 'validation paradigms')

# plot 4 images per category
# create new plot window
enum = []
for i in range(num_classes):
    enum.append([])
for i, j in enumerate(y_train):
    enum[j].append(x_train[i])

for i in range(num_classes):
    class_to_demonstrate = label_names[i]
    x = enum[i]
    np.random.shuffle(x)
    plt.figure()
    plt.suptitle(class_to_demonstrate)
    plt.subplot(221)
    plt.imshow(x[0])
    plt.subplot(222)
    plt.imshow(x[1])
    plt.subplot(223)
    plt.imshow(x[2])
    plt.subplot(224)
    plt.imshow(x[3])
    plt.show()

# no need to reshape anything. Already shaped
# x_train = x_train.reshape(len(x_train), 32, 32, 3)
# x_valid = x_valid.reshape(len(x_valid), 32, 32, 3)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

# here we define and load the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# print model summary
model.summary()

# fit model parameters, given a set of training data
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_valid, y_valid))

# saving the trained model
model_name = '../OutputData/CNN_1.h5'
model.save(model_name)

print('CNN topology setup completed')
