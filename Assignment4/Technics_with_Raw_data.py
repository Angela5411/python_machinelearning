import time
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from mnist import MNIST
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


# the data, split between train and test sets
mndata = MNIST('C:\Program Files\Python37\Lib\site-packages\mnist\samples')
# X_test, y_test = mndata.load_testing()
# X_test, y_test = np.array(X_test), np.array(y_test)
X_train, y_train = mndata.load_training()
X_train, y_train = np.array(X_train), np.array(y_train)
# print(np.shape(X_train))

# do not forget the validate set
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.1, random_state=1)
#
# # reshaping the data to appropriate tensor format
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
# X_validate = X_validate.reshape(X_validate.shape[0], 28, 28, 1)
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# print(np.shape(X_train))

params = {'quantile': .3,
            'eps': .3,
            'damping': .9,
            'preference': -200,
            'n_neighbors': 10,
            'n_clusters': 10,
            'min_samples': 20,
            'xi': 0.05,
            'min_cluster_size': 0.1}


X= X_train

# normalize dataset for easier parameter selection
X = StandardScaler().fit_transform(X)

# ============
# Create cluster objects
# ============

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
    try:
        # saving the trained model
        joblib.dump(algorithm, './OutputData/'+name+'.pkl')

    except:
        print('error while saving ',name)
        continue


    t1 = time.time()
    #calculate the performance scores
    print(' Just finished with ', name, 'in {:.3f}'.format(t1-t0), 'seconds.')


# fortwnw standard scalR 0-1 3 TEXN PREDICT print scores neurona auto encoder kai 3ana ta idia
