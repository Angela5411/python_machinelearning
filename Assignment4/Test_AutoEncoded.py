 # 10. Χρησιμοποιώντας τα  αποτελέσματα  του  αλγορίθμου,
# και  γραφικές  παραστάσεις  που  θα φτιάξετε στο excel,
# θα συντάξετε μια έκθεση στην οποία θα παρουσιάζετε
# τα συμπεράσματα σας, θα κάνετε συγκριτικές αξιολογήσεις και
# θα προτείνετε ποια είναι η καλύτερη δυνατή τεχνική
# για την συγκεκριμένη περίπτωση.

#define a performance function
from itertools import islice, cycle
import matplotlib.pyplot as plt
import joblib
import numpy as np
import keras
from mnist import MNIST
from keras import backend as K
from numpy import unique
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
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
# Davies–Bouldin index, Dunn index,Silhouette coefficient
        # Purity, Rand index, F-measure, Jaccard index, Dice index,Fowlkes–Mallows index,  mutual information ,Confusion matrix
        # Hopkins statistic,
        #Elbow


try:
    # saving the trained model
    model_name = './OutputData/Autoencoder.h5'
    # loading a trained model & use it over test data
    model = keras.models.load_model(model_name)
except:
    print('error while reading model')
else:
    print('successfully read model')

mndata = MNIST('C:\Program Files\Python37\Lib\site-packages\mnist\samples')
# X_test, y_test = mndata.load_testing()
# X_test, y_test = np.array(X_test), np.array(y_test)
X_train, y_train = mndata.load_training()
X_train, y_train = np.array(X_train), np.array(y_train)

X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.1, random_state=1)
Xb,yb=X_train,y_train
Xb = Xb.reshape(Xb.shape[0], 28, 28, 1)

# Extract the encoder
encoder = K.function([model.layers[0].input], [model.layers[4].output])

# Encode the training set
X = encoder([Xb])[0].reshape(-1, 7*7*7)
print(np.shape(X))

clustering_algorithms = (
    ('SpectralClust', None),
    ('MiBatchKMeans', None),
    ('DBSCAN', None)
)

# Fitting testing dataset
for name, algorithm in clustering_algorithms:
    algorithm = joblib.load('./OutputData/'+name+'AE.pkl')

    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)

    s,ch,db,my1= performance_score(X, y_pred,y_train)
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

    pre_test = (precision_score(yb, y_pred, average='weighted', labels=unique(y_pred)))
    rec_test = (recall_score(yb, y_pred, average='weighted', labels=unique(y_pred)))
    f1_test = (f1_score(yb, y_pred, average='weighted', labels=unique(y_pred)))
    acc_test = (accuracy_score(yb, y_pred))
    print('precision_score', pre_test, 'recall_score', rec_test, 'f1_score', f1_test, 'accuracy_score', acc_test)
