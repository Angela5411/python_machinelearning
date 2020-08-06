# 7. Θα	υπολογίζει τους δείκτες απόδοσης που παρουσιάστηκαν στο	εργαστήριο και έναν ακόμα της επιλογή σας.
# 8. Θα παρουσιάζει ενδεικτικά αποτελέσματα
# ομαδοποίησης για τυχαίες εικόνες.
# 9. Τα βήματα 6, 7 και 8 θα εκτελεστούν χρησιμοποιώντας
# α) τις τιμές των pixel των εικόνων
# (κανονικοποιημένες  στο  [0,1]  και
# β)  τις  τιμές  των  εικόνων  που  παράγει
# το  κομμάτι του encoder στο CNN που κατασκευάσατε.
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
from numpy import unique
from sklearn import metrics
from mnist import MNIST
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


mndata = MNIST('C:\Program Files\Python37\Lib\site-packages\mnist\samples')
# X_test, y_test = mndata.load_testing()
# X_test, y_test = np.array(X_test), np.array(y_test)
X_train, y_train = mndata.load_training()
X_train, y_train = np.array(X_train), np.array(y_train)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

X,y=X_train,y_train
# normalize dataset for easier parameter selection
X = StandardScaler().fit_transform(X)


clustering_algorithms = (
    ('SpectralClust', None),
    ('MiBatchKMeans', None),
    ('DBSCAN', None)
)

plot_num=1
for name, algorithm in clustering_algorithms:
    algorithm = joblib.load('./OutputData/'+name+'.pkl')

    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)
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

    pre_test = (precision_score(y, y_pred, average='weighted', labels=unique(y_pred)))
    rec_test = (recall_score(y, y_pred, average='weighted', labels=unique(y_pred)))
    f1_test = (f1_score(y, y_pred, average='weighted', labels=unique(y_pred)))
    acc_test = (accuracy_score(y, y_pred))
    print('precision_score', pre_test, 'recall_score', rec_test, 'f1_score', f1_test, 'accuracy_score', acc_test)
