#TestTheModel
# Sfyridaki Angeliki cs151036
# Mhxanikwn plhroforikhs
# cs151036@uniwa.gr
# Part 1

import keras
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sn
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

batch_size = 128
num_classes = 10

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
    fileNameToLoad = '../InputData/dataset/test_batch'  # define the txt file to pass the examples
    listX, listY = (unpickle(fileNameToLoad))
except:
    print('error while reading data')
    exit(2)
else:
    print('successfully read data')


x_test = np.array(listX)
y_test= np.array(listY)

try:
    # saving the trained model
    model_name = '../OutputData/CNN_1.h5'
    # loading a trained model & use it over test data
    loaded_model = keras.models.load_model(model_name)
except:
    print('error while reading model')
else:
    print('successfully read model')




y_test_predictions_vectorized = loaded_model.predict(x_test)#shows posibilities
y_test_predictions = np.argmax(y_test_predictions_vectorized, axis=1)
print('test predicted')

cm=confusion_matrix(y_test,y_test_predictions)#tn, fp, fn, tp
df_cm = pd.DataFrame(cm, label_names,label_names)
plt.figure(figsize = (12,12))
sn.set(font_scale=1.2)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 13}, cmap='inferno')# font size
plt.show()

pre_test = (precision_score(y_test, y_test_predictions, average='macro'))
rec_test = (recall_score(y_test, y_test_predictions, average='macro'))
f1_test = (f1_score(y_test, y_test_predictions, average='macro'))
acc_test = (accuracy_score(y_test, y_test_predictions))
print('precision_score',pre_test,'recall_score',rec_test,'f1_score',f1_test,'accuracy_score',acc_test)

#create new plot window
enum = []
for i in range(num_classes):
    enum.append([])
for i, j in enumerate(y_test_predictions):
    enum[j].append(x_test[i])

for i in range(num_classes):
    x = enum[i]
    class_to_demonstrate = label_names[i]
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

