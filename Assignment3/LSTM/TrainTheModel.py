#1. TrainTheModel.
# Sfyridaki Angeliki cs151036
# Mhxanikwn plhroforikhs
# cs151036@uniwa.gr
# Part 2
from random import seed
from random import randint
from numpy import array
from math import ceil
from math import log10
from numpy import argmax
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector

# Στην οθόνη  πρέπει  να  εμφανίζονται  όλα  τα  σχετικά  μηνύματα
# που  πληροφορούν  για  την κατάσταση και το σημείο που βρίσκεται ο κώδικας.

# generate lists of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
    X, y = list(), list()
    for i in range(n_examples):#1000 tuxaia zeugaria
        in_pattern = [randint(0, largest) for _ in range(n_numbers)]
        out_pattern = sum(in_pattern)#sums the random numbers
        X.append(in_pattern)
        y.append(out_pattern)
    return X, y

def random_sub_pairs(n_examples, n_numbers, largest):
    X, y = list(), list()
    for i in range(n_examples):#1000 tuxaia zeugaria
        in_pattern = [randint(0, largest) for _ in range(n_numbers)]
        out_pattern = in_pattern[0]-sum(in_pattern[1:])#subtracts the random numbers 10-9-9-10-7
        X.append(in_pattern)
        y.append(out_pattern)
    return X, y

# convert data to strings
def sum_to_string(X, y, n_numbers, largest):
    max_length = n_numbers * ceil(log10(largest + 1)) + n_numbers - 1#ceil: The smallest integer greater than or equal to the given number.
    Xstr = list()
    for pattern in X:
        strp = '+'.join([str(n) for n in pattern])
        strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp #pros8etei kena wste a8roistika na exei 5 yhfia
        Xstr.append(strp)
    max_length = ceil(log10(n_numbers * (largest + 1)))
    ystr = list()
    for pattern in y:
        strp = str(pattern)
        strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp #pros8etei kena wste a8ristoika n exei 3 yhfia
        ystr.append(strp)
    return Xstr, ystr

# convert data to strings
def sub_to_string(X, y, n_numbers, largest):
    max_length = n_numbers * ceil(log10(largest + 1)) + n_numbers - 1
    Xstr = list()
    for pattern in X:
        strp = '-'.join([str(n) for n in pattern])
        strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp #pros8etei kena wste a8roistika na exei 5 yhfia
        Xstr.append(strp)
    max_length = ceil(log10(n_numbers * (largest + 1)))
    ystr = list()
    for pattern in y:
        strp = str(pattern)
        strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp #pros8etei kena wste a8ristoika n exei 3 yhfia
        ystr.append(strp)
    return Xstr, ystr

# integer encode strings
#translates every char to intcode px + --> 10
#5*[], 2*[]
def integer_encode(X, y, alphabet):
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for pattern in X:
        integer_encoded = [char_to_int[char] for char in pattern]
        Xenc.append(integer_encoded)
    yenc = list()
    for pattern in y:
        integer_encoded = [char_to_int[char] for char in pattern]
        yenc.append(integer_encoded)
    return Xenc, yenc


# one hot encode
# translates every char into a [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
def one_hot_encode(X, y, max_int):
    Xenc = list()
    for seq in X:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Xenc.append(pattern)
    yenc = list()
    for seq in y:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        yenc.append(pattern)
    return Xenc, yenc


# generate an encoded dataset
def generate_data(n_samples, n_numbers, largest, alphabet):
    # generate pairs
    X, y = random_sum_pairs(int(n_samples/2), n_numbers, largest)
    #print(X,'\n',y,'\n',len(y))
    # convert to strings
    X, y = sum_to_string(X, y, n_numbers, largest)
    #print(X,'\n',y,'\n',len(y))
    # generate pairs
    x, Y = random_sub_pairs(int(n_samples/2), n_numbers, largest)
    #print(x, '\n', Y,'\n',len(Y))
    # convert to strings
    x, Y = sub_to_string(x, Y, n_numbers, largest)
    #print(x, '\n', Y,'\n',len(Y))
    # integer encode    extend 2 lists
    X, y = integer_encode(x+X, Y+y, alphabet)
    #print(X,'\n',y,'\n',len(y))
    # one hot encode
    X, y = one_hot_encode(X, y, len(alphabet))
    #print(X,'\n',y)
    # return as numpy arrays
    X, y = array(X), array(y)
    return X, y



# define dataset
seed(1)
n_samples = 10000
n_numbers = 2
largest = 99
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ', '-']
n_chars = len(alphabet)
#integer largest has ceil(log10(largest + 1)) digits.
n_in_seq_length = n_numbers * ceil(log10(largest + 1)) + n_numbers - 1  #5
n_out_seq_length = ceil(log10(n_numbers * (largest + 1)))       #2

# define LSTM configuration
n_batch = 10
n_epoch = 40
#X, y = generate_data(n_samples, n_numbers, largest, alphabet)


# create LSTM
model = Sequential()
model.add(LSTM(100, input_shape=(n_in_seq_length, n_chars)))
model.add(RepeatVector(n_out_seq_length))
model.add(LSTM(50, return_sequences=True))
model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())



# train LSTM
for i in range(n_epoch):
    X, y = generate_data(n_samples, n_numbers, largest, alphabet)
    print(i)
    model.fit(X, y, epochs=1, batch_size=n_batch)

# saving the trained model
model_name = '../OutputData/LSTM_1.h5'
model.save(model_name)

print('LSTM topology setup completed')
