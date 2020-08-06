#Sfyridaki Angeliki cs151036
#Mhxanikwn plhroforikhs
#cs151036@uniwa.gr
#Part 1


import pandas as pd # excel reading
import sklearn # we need this for the classifiers
import numpy as np # mathematical operators
import keras # artificial neural networks (it uses TensorFlow)
import matplotlib.pyplot as plt
import xlrd
#import secondary functions that will be used very frequent
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler

#where 0:survived 1:not survived
#so tp : not survived tn:survived
# def scores(true,pred):
#     classes=np.unique(true)
#     tp=0;    fp=0;    tn=0;    fn=0
#     for i in range(len(pred)):
#         if true[i]==pred[i]==1:
#             tp+=1
#         if pred[i]==1 and true[i]!=pred[i]:
#             fp+=1
#         if true[i]==pred[i]==0:
#             tn+=1
#         if pred[i]==0 and true[i]!=pred[i]:
#             fn+=1
#
#     return tp,fp,tn,fn

AcceptanceList=[]
def CheckBest(classname,recall,tn,fp):
    if recall>=0.62 and tn/(tn+fp)>=0.7:
        AcceptanceList.append(classname)



def MakeScoreLists1(className,y_train,y_pred_train):
    transpose[0].append(className)
    transpose[1].append('train set')
    transpose[2].append(len(X_train))

    tmpCount = sum(y_train == 1)
    transpose[3].append(tmpCount)
    tn, fp, fn, tp=confusion_matrix(y_train, y_pred_train).ravel()
    # tp, fp, tn, fn = scores(y_train, y_pred_train)

    transpose[4].append(tp)
    transpose[5].append(tn)
    transpose[6].append(fp)
    transpose[7].append(fn)

    pre_train=(precision_score(y_train, y_pred_train, average='macro'))
    transpose[8].append(pre_train)
    rec_train=(recall_score(y_train, y_pred_train, average='macro'))
    transpose[9].append(rec_train)
    f1_train=(f1_score(y_train, y_pred_train, average='macro'))
    transpose[10].append(f1_train)
    acc_train=(accuracy_score(y_train, y_pred_train))
    transpose[11].append(acc_train)

def MakeScoreLists2(className,y_test,y_pred_test):
    transpose[0].append(className)
    transpose[1].append('test set')
    transpose[2].append(len(X_test))
    tmpCount = sum(y_test == 1)
    transpose[3].append(tmpCount)

    tn, fp, fn, tp=confusion_matrix(y_test, y_pred_test).ravel()
    # tp, fp, tn, fn = scores(y_test, y_pred_test)
    transpose[4].append(tp)
    transpose[5].append(tn)
    transpose[6].append(fp)
    transpose[7].append(fn)

    pre_test=(precision_score(y_test, y_pred_test, average='macro'))
    transpose[8].append(pre_test)
    rec_test=(recall_score(y_test, y_pred_test, average='macro'))
    transpose[9].append(rec_test)
    f1_test=(f1_score(y_test, y_pred_test, average='macro'))
    transpose[10].append(f1_test)
    acc_test=(accuracy_score(y_test, y_pred_test))
    transpose[11].append(acc_test)
    CheckBest(className,rec_test,tn,fp)


#######################################################################################################################
#---------------------------------- 1. -----------------------------------------------------------------------------------
#######################################################################################################################


# read the data
fileName = './InputData/Dataset2Use_Assignment2.xlsx' # you may add the full path here
sheetName = 'Total'
try:
    # Confirm file exists.
    sheetValues = pd.read_excel(fileName, sheetName)
    print(' .. successful parsing of file:', fileName)
   # print("Column headings:")
   # print(sheetValues.columns)
except FileNotFoundError:
    print(FileNotFoundError)

#fisrt get only the numeric values; i.e. ignore the last 2 columns, and convert it to ndarray
inputData = sheetValues[sheetValues.columns[:-2]].values    #ignore columne M
#now convert the categorical values to unique class id and save the name-toid match
outputData = sheetValues[sheetValues.columns[-2]]
outputData, levels = pd.factorize(outputData)
#outputData: array([0 0 0 0 0...])     levels: Int64Index([1,2])   outputData->levels      0->1        1->2

print(' .. we have', inputData.shape[0], 'available paradigms.')
print(' .. each paradigm has', inputData.shape[1], 'features')

print(' ... the distribution for the available class lebels is:')
for classIdx in range(0, len(np.unique(outputData))):
    tmpCount = sum(outputData == classIdx)
    tmpPercentage = tmpCount/len(outputData)
    print(' .. class', str(classIdx), 'has', str(tmpCount), 'instances', '(', '{:.2f}'.format(tmpPercentage), '%)')


#inputData.shape[0]: all paradigms for training
##inputData.shape[1]: Num of features(A to K)
#tmpCount[1]=No of occurasies per category
#tmpPercentage[1]=Propabilities Pi/N
#
# #check the correlation between features
# feature_names = ['mass', 'width', 'height', 'color_score']
# X = pd.DataFrame(inputData)
# y = outputData
# cmap = cm.get_cmap('gnuplot')
# scatter = scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
# plt.suptitle('Scatter-matrix for each input variable')
# plt.savefig('fruits_scatter_matrix')
# plt.show()


#now create train and test sets
X_train, X_test, y_train, y_test = train_test_split(inputData, outputData, random_state=0)
scaler = MinMaxScaler() #Constructing normalization scaling
X_train = scaler.fit_transform(X_train) #computes minimum and maximum to be used for later scaling and then transforms x_train
X_test = scaler.transform(X_test)   #scaling features of X_test according to feature_range




#######################################################################################################################
#---------------------------------- 2. --------------------------------------------------------------------------------
#######################################################################################################################
#now the models


transpose = []
for i in range(0, 12):
    transpose.append([])

#---------------------------- Logistic	Regression --------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train) #fit the model using the training data
#now check for both train and test data, how well the model learned the patterns

y_pred_train = logreg.predict(X_train)
y_pred_test = logreg.predict(X_test)

#calculate the scores
MakeScoreLists1('LogisticRegression',y_train,y_pred_train)
MakeScoreLists2('LogisticRegression',y_test,y_pred_test)



#---------------------------- Decision	Trees --------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train) #fit the model using the training data
#now check for both train and test data, how well the model learned the patterns
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
#calculate the scores
MakeScoreLists1('DecisionTree',y_train,y_pred_train)
MakeScoreLists2('DecisionTree',y_test,y_pred_test)





#---------------------------- k-Nearest	Neighbors --------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train) #fit the model using the training data
#now check for both train and test data, how well the model learned the patterns
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)
#calculate the scores
MakeScoreLists1('KNeighbors',y_train,y_pred_train)
MakeScoreLists2('KNeighbors',y_test,y_pred_test)



#----------------------------Linear	Discriminant	Analysis --------------------------------------------------------------------

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train) #fit the model using the training data
#now check for both train and test data, how well the model learned the patterns
y_pred_train = lda.predict(X_train)
y_pred_test = lda.predict(X_test)
#calculate the scores
MakeScoreLists1('LinearDiscriminantAnalysis',y_train,y_pred_train)
MakeScoreLists2('LinearDiscriminantAnalysis',y_test,y_pred_test)





#----------------------------Naïve	Bayes--------------------------------------------------------------------

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train) #fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = gnb.predict(X_train)
y_pred_test = gnb.predict(X_test)
# calculate the scores
MakeScoreLists1('naive_bayes',y_train,y_pred_train)
MakeScoreLists2('naive_bayes',y_test,y_pred_test)



#---------------------------- Support	Vector	Machines --------------------------------------------------------------------

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train) #fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
y_pred_train = gnb.predict(X_train)
y_pred_test = gnb.predict(X_test)
# calculate the scores
MakeScoreLists1('SVC',y_train,y_pred_train)
MakeScoreLists2('SVC',y_test,y_pred_test)



# #---------------------------- Neural Networks --------------------------------------------------------------------

#finally, build a feed forward neural network using keras
CustomModel = keras.models.Sequential()
CustomModel.add(keras.layers.Dense(16, input_dim=X_train.shape[1], activation='relu'))
CustomModel.add(keras.layers.Dense(2, activation='softmax'))
#display the architecture
#CustomModel.summary()
#compile model using accuracy to measure model performance
CustomModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model
CustomModel.fit(X_train, keras.utils.np_utils.to_categorical(y_train), epochs=100, verbose=False)
y_pred_train = CustomModel.predict_classes(X_train)
y_pred_test = CustomModel.predict_classes(X_test)

MakeScoreLists1('neural network',y_train,y_pred_train)
MakeScoreLists2('neural network',y_test,y_pred_test)


#save the results to a txt file (inputs, outputs1, outputs2)
fileNameToSaveAndLoad = './OutputData/Results.txt' #define the txt file to pass the examples
headerValues = ['Classifier Name','Training or test set','Number of training samples','Number of non-healthy companies in training sample',
                'TP','TN','FP','FN','Precision','Recall','F1 score','Accuracy'] #column headers



valuesToSave={}
for i,obj in enumerate(transpose):
     valuesToSave[headerValues[i]]=obj
try:
    df = pd.DataFrame(valuesToSave)
    writer = pd.ExcelWriter('./OutputData/Results.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

except FileNotFoundError:
    print('error while writing data', FileNotFoundError)
else:
    print('successfully writen')


#1. Το	μοντέλο πρέπει	να	βρίσκει με	ποσοστό επιτυχίας τουλάχιστον 62% τις	εταιρείες που	θα	πτωχεύσουν.
#p=N(brhka oti ptwxeusan apo autes p ptoxeusan)/N(ptwxeusan)=tp/(tp+fn))=recall
#2. Το	μοντέλο πρέπει	να	βρίσκει με	ποσοστό επιτυχίας	τουλάχιστον 70% τις	εταιρείες που	δεν θα	πτωχεύσουν
#p=N(brhka oti den ptoxeusan apo autes pou den ptoxeusan)/N(den ptoxeusan)=tn/(tn+fp))
if len(AcceptanceList)!=0:
    print(AcceptanceList)
else:
    print('no available model')