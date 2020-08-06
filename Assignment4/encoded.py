
import numpy as np
import time
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from mnist import MNIST
import keras
from keras import backend as K
from sklearn import metrics
from sklearn import cluster
import matplotlib.pyplot as plt
import seaborn as sns


def performance_score(input_values, cluster_indexes,true_indexes):

    try:
        silh_score = metrics.silhouette_score(input_values, cluster_indexes)
        print(' .. Silhouette Coefficient score is {:.2f}'.format(silh_score))
        print( ' ... -1: incorrect, 0: overlapping, +1: highly dense clusters.')
    except:
        print(' .. Warning: could not calculate Silhouette Coefficient score.')
        silh_score = -999

    try:
        ch_score = metrics.calinski_harabasz_score(input_values, cluster_indexes)
        print(' .. Calinski-Harabasz Index score is {:.2f}'.format(ch_score))
        print(' ... Higher the value better the clusters.')
    except:
        print(' .. Warning: could not calculate Calinski-Harabasz Index score.')
        ch_score = -999

    try:
        db_score = metrics.davies_bouldin_score(input_values, cluster_indexes)
        print(' .. Davies-Bouldin Index score is {:.2f}'.format(db_score))
        print(' ... 0: Lowest possible value, good partitioning.')
    except:
        print(' .. Warning: could not calculate Davies-Bouldin Index Index score.')
        db_score = -999

    try:
        ars = metrics.adjusted_rand_score(true_indexes, cluster_indexes)
        print(' .. adjusted rand score is {:.2f}'.format(ars))
        print(' ... Perfect labeling is scored 1.0 Bounded range [-1, 1]')
    except:
        print(' .. Warning: could not calculate adjusted rand score.')
        ars = -999

    return silh_score, ch_score, db_score, ars


batch_size = 128
num_classes = 10
epochs = 2

# input image dimensions
img_rows, img_cols = 28, 28

try:
    # saving the trained model
    model_name = './OutputData/Autoencoder.h5'
    # loading a trained model & use it over test data
    model = keras.models.load_model(model_name)
except:
    print('error while reading model')
else:
    print('successfully read model')

# the data, split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
mndata = MNIST('C:\Program Files\Python37\Lib\site-packages\mnist\samples')
# X_test, y_test = mndata.load_testing()
# X_test, y_test = np.array(X_test), np.array(y_test)
X_train, y_train = mndata.load_training()
X_train, y_train = np.array(X_train), np.array(y_train)

# do not forget the validate set
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

# reshaping the data to appropriate tensor format
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
# X_validate = X_validate.reshape(X_validate.shape[0], 28, 28, 1)
#X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# print(np.shape(X_train))
# Fitting testing dataset
Xb,yb=X_train,y_train
restored_testing_dataset = model.predict(Xb)
print('restored',np.shape(restored_testing_dataset))
# Observe the reconstructed image quality
plt.figure(figsize=(20,5))
for i in range(10):
    index = yb.tolist().index(i)
    plt.subplot(2, 10, i+1)
    plt.imshow(Xb[index].reshape((28,28)))
    plt.gray()
    plt.subplot(2, 10, i+11)
    plt.imshow(restored_testing_dataset[index].reshape((28,28)))
    plt.gray()

# Extract the encoder
encoder = K.function([model.layers[0].input], [model.layers[4].output])
# Encode the training set
X = encoder([Xb])[0].reshape(-1, 7*7*7)
print('enc im',np.shape(X))


params = {'quantile': .3,
            'eps': .3,
            'damping': .9,
            'preference': -200,
            'n_neighbors': 10,
            'n_clusters': 10,
            'min_samples': 20,
            'xi': 0.05,
            'min_cluster_size': 0.1}

spectral = cluster.SpectralClustering(
    n_clusters=params['n_clusters'], eigen_solver='arpack',
    affinity="nearest_neighbors")
two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
dbscan = cluster.DBSCAN(eps=params['eps'])

clustering_algorithms = (
    ('SpectralClust', spectral),
    ('MiBatchKMeans', two_means),
    ('DBSCAN', dbscan)
)

for name, algorithm in clustering_algorithms:
    t0 = time.time()
    print('X bef fit', np.shape(X))
    # catch warnings related to kneighbors_graph
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="the number of connected components of the " +
            "connectivity matrix is [0-9]{1,2}" +
            " > 1. Completing it to avoid stopping the tree early.",
            category=UserWarning)
        warnings.filterwarnings(
            "ignore",
            message="Graph is not fully connected, spectral embedding" +
            " may not work as expected.",
            category=UserWarning)
        algorithm.fit(X)
    # try:
    #     # saving the trained model
    #     #joblib.dump(algorithm, './OutputData/'+name+'AE.pkl')
    #
    # except:
    #     print('error while saving ',name)
    #     continue


    t1 = time.time()
    #calculate the performance scores
    print(' Just finished with ', name, 'in {:.3f}'.format(t1-t0), 'seconds.')

    # Fitting testing dataset
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)

    print('pred', np.shape(y_pred))

    s,ch,db,my1= performance_score(X, y_pred,yb)
    print(np.shape(X),np.shape(y_pred),np.shape(yb))
    print(s,ch,db,my1)

    cm = confusion_matrix(yb, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    plt.show()

    # Plot the actual pictures grouped by clustering
    fig = plt.figure(figsize=(20,10))
    for r in range(10):
        cluster = cm[r].argmax()
        for c, val in enumerate(Xb[y_pred == cluster][0:4]):
            fig.add_subplot(10, 4, 4 * r + c + 1)
            plt.imshow(val.reshape((28, 28)))
            plt.gray()
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('cluster: ' + str(cluster))
            plt.ylabel('digit: ' + str(r))
    plt.show()
    plt.pause(0.1)
