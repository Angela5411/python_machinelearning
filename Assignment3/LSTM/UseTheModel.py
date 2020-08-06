# Sfyridaki Angeliki cs151036
# Mhxanikwn plhroforikhs
# cs151036@uniwa.gr
# Part 2

from random import randint
from keras.engine.saving import load_model
from numpy import array, unique
from math import ceil
from math import log10
from numpy import argmax
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import re

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

def read_data(X,symbol,y=[]):
    # convert to strings
    max_length = n_numbers * ceil(log10(largest + 1)) + n_numbers - 1  # ceil: The smallest integer greater than or equal to the given number.
    Xstr = list()
    if symbol is '+':
        for pattern in X:
            strp = '+'.join([n for n in pattern])
            strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp  # pros8etei kena wste a8roistika na exei 5 yhfia
            Xstr.append(strp)
        # integer encode    extend 2 lists
    else:
        # convert to strings
        for pattern in X:
            strp = '-'.join([n for n in pattern])
            strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp  # pros8etei kena wste a8roistika na exei 5 yhfia
            Xstr.append(strp)
    X=Xstr
    #print('sum_to_string\n',X, '\n', len(y))

    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for pattern in X:
        integer_encoded = [char_to_int[char] for char in pattern]
        Xenc.append(integer_encoded)
    X=Xenc
    #print('integer_encode\n',X)

     # one hot encode
    Xenc = list()
    for seq in X:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(len(alphabet))]
            vector[index] = 1
            pattern.append(vector)
        Xenc.append(pattern)
    X=Xenc
    #print('one_hot_encode\n',X,'\n',y)

    # return as numpy arrays
    X= array(X)
    result = model.predict(X, batch_size=n_batch, verbose=0)
    predicted = [invert(x, alphabet) for x in result]
    return predicted


# invert encoding
def invert(seq, alphabet):
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    strings = list()
    for pattern in seq:
        string = int_to_char[argmax(pattern)]
        strings.append(string)
    return ''.join(strings)

def insert():
    input_str = input('give some input or press 0 for exit')
    while input_str is not '0':
        input_str = input_str.strip()
        x = re.search('^([0-9]+)([\+\-][0-9]+)?', input_str)
        if x != None:
            x = x.string
            digits = re.findall('[0-9]+', x)
            symbols = re.findall('[\+\-]', x)

            if len(symbols) == len(digits):
                del symbols[-1]

            for i in digits:
                if int(i) > largest or int(i)< 0:
                    print('numbers out of range 1-99')
                    x = None;
                    break

            if x is not None:
                y = [[digits[0], 0]]
                if len(digits) > 1:
                    y[0][1] = digits[1]
                    y[0][0] = read_data(y, symbols[0])[0]
                    del symbols[0]
                    del digits[0]
                print(x, '=', y[0][0])

        input_str = input('give some input or press 0 for exit')
    return


try:
    # saving the trained model
    model_name = '../OutputData/LSTM_1.h5'
    # loading a trained model & use it over test data
    model = load_model(model_name)
except:
    print('error while reading model')
else:
    print('successfully read model')



n_samples = 3000
n_numbers = 2
largest = 99
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ', '-']
n_chars = len(alphabet)
n_batch = 10
#integer largest has ceil(log10(largest + 1)) digits.
n_in_seq_length = n_numbers * ceil(log10(largest + 1)) + n_numbers - 1  #5
n_out_seq_length = ceil(log10(n_numbers * (largest + 1)))       #2


# evaluate on some new patterns
X, y = generate_data(n_samples, n_numbers, largest, alphabet)
result = model.predict(X, batch_size=n_batch, verbose=0)
# calculate error
string=[invert(x, alphabet) for x in X]
expected = [invert(x, alphabet) for x in y]
predicted = [invert(x, alphabet) for x in result]

pre_test = (precision_score(expected, predicted,average='weighted', labels=unique(predicted)))
rec_test = (recall_score(expected, predicted, average='weighted', labels=unique(predicted)))
f1_test = (f1_score(expected, predicted, average='weighted', labels=unique(predicted)))
acc_test = (accuracy_score(expected, predicted))
print('precision_score',pre_test,'recall_score',rec_test,'f1_score',f1_test,'accuracy_score',acc_test)

# show some examples
print('show some examples')
for i in range(20):
    print('%s, =%s' % (string[i], predicted[i]))

insert()