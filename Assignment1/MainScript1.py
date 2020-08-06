#Sfyridaki Angeliki cs151036
#Mhxanikwn plhroforikhs
#cs151036@uniwa.gr
#Part 1

#necessary python packages
import numpy as np                 #mathematical operations
import scipy.optimize as opt       #curve fitting tools
import matplotlib.pyplot as plt    #plotting tools

#returns a sorted list of N numbers
#N=amount of random values  low=starting value point    high=ending value point
def Answer1_1(low,high,N):
    answer = np.random.uniform(low, high, N) #N values  from low to high
    return sorted(answer)


#answer1_2

#y=-(x-3)^2+2
def custFunc1(x):
    k=[]
    for i in x:
        k=k+[-(i-3)**2+2]
    return k

#y=sin(x)+(1/e(x))
def custFunc2(x):
    k=[]
    for i in x:
        k=k+[np.sin(i)+(1/np.exp(i))]
    return k

low,high,N=0.,33.,100
rundomNums= Answer1_1(low,high,N)

#answer1_3
Func1Array= custFunc1(rundomNums)
Func2Array= custFunc2(rundomNums)

#answer1_4
fig = plt.figure(tight_layout=True)

ax= fig.add_subplot(211)
ax.plot(rundomNums, Func1Array, 'r*', label='function1')  #this is the basic plot function
ax.set_xlabel('InputValue')
ax.set_ylabel('Func1Output')
plt.title('exponential model')


ax2=fig.add_subplot(212)
ax2.plot(rundomNums, Func2Array, 'b+', label='function2')
ax2.set_xlabel('InputValue')
ax2.set_ylabel('Func2Output')
plt.title('trigonometrical model')

plt.show()


#answer1_5
#save the results to a txt file (inputs, outputs1, outputs2)
fileNameToSaveAndLoad = './OutputData/Results.txt' #define the txt file to pass the examples
headerValues = 'InputValue Func1Output Func2Output' #column headers
valuesToSave = np.column_stack([rundomNums, Func1Array, Func2Array])
try:
    np.savetxt(fileNameToSaveAndLoad, valuesToSave, fmt="%.2f", header=headerValues)
except:
    print('error while writing data')
else:
    print('successfully writen')

