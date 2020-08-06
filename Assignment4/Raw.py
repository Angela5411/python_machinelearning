import time
import warnings
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from mnist import MNIST
from sklearn import cluster, metrics
from sklearn.preprocessing import StandardScaler
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


# the data, split between train and test sets
mndata = MNIST('C:\Program Files\Python37\Lib\site-packages\mnist\samples')
# X_test, y_test = mndata.load_testing()
# X_test, y_test = np.array(X_test), np.array(y_test)
X_train, y_train = mndata.load_training()
X_train, y_train = np.array(X_train), np.array(y_train)
# print(np.shape(X_train))

# do not forget the validate set
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.1, random_state=1)
X= X_train
print('X', np.shape(X))

# normalize dataset for easier parameter selection
X = StandardScaler().fit_transform(X)

params = {'quantile': .3,
            'eps': .3,
            'damping': .9,
            'preference': -200,
            'n_neighbors': 10,
            'n_clusters': 10,
            'min_samples': 20,
            'xi': 0.05,
            'min_cluster_size': 0.1}


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
    X = X.reshape(X.shape[0], 28*28)
    print('X SS', np.shape(X))

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
    #     joblib.dump(algorithm, './OutputData/'+name+'.pkl')
    #
    # except:
    #     print('error while saving ',name)
    #     continue


    t1 = time.time()
    #calculate the performance scores
    print(' Just finished with ', name, 'in {:.3f}'.format(t1-t0), 'seconds.')


    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)
    print('ypred', np.shape(y_pred))

    s,ch,db,my1= performance_score(X, y_pred,y_train)

    X = X.reshape(X.shape[0], 28, 28, 1)
    print(s,ch,db,my1)

    cm = confusion_matrix(y_train, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    plt.show()
    plt.pause(0.1)

    # Plot the actual pictures grouped by clustering
    fig = plt.figure(figsize=(20,10))
    for r in range(10):
        cluster = cm[r].argmax()
        for c, val in enumerate(X_train[y_pred == cluster][0:4]):
            fig.add_subplot(10, 4, 4 * r + c + 1)
            plt.imshow(val.reshape((28, 28)))
            plt.gray()
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('cluster: ' + str(cluster))
            plt.ylabel('digit: ' + str(r))
    plt.show()
    plt.pause(0.1)



# fortwnw standard scalR 0-1 3 TEXN PREDICT print scores neurona auto encoder kai 3ana ta idia
