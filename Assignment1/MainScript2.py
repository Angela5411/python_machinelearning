#Sfyridaki Angeliki cs151036
#Mhxanikwn plhroforikhs
#cs151036@uniwa.gr
#Part2

#necessary python packages
import numpy as np                 #mathematical operations
import scipy.optimize as opt       #curve fitting tools
import matplotlib.pyplot as plt    #plotting tools


#y=a*x^3+b*x^2+c*x+d
def custFunc3(x,a,b,c,d):
    return a* x**3 + b* x**2 + c*x + d


#y=a*x^4+b*x^3+c*x^2+d*x+e
def custFunc4(x,a,b,c,d,e):
    return a*x**4+b*x**3+c*x**2+d*x+e

#y=a*x^4+b*x^3+c*x^2+d*x+e
def custFunc8(x,a,b,c,d,e,f,g,h,j):
    return a*x**8+b*x**7+c*x**6+d*x**5+e*x**4+f*x**3+g*x**2+h*x+j

def custFuncC(x,a,b,c):
    return a*np.sin(b*x+c)


#Simple error
def dif(actual, predicted):
    k=[]
    for i,o in enumerate(actual):
        k.append(o - predicted[i])
    return k

#Mean Absolute Error
def mae(actual, predicted):
    return np.mean(np.abs(dif(actual,predicted)))

#Mean Squared Error
def mse(actual, predicted):
    return np.mean(np.square(dif(actual, predicted)))

#answer2_1
fileNameToLoad = './OutputData/Results.txt' #define the txt file to pass the examples
#creates a big 100 elemenents list where each element is a sublist of 3 numbers(InputValue Func1Output Func2Output)
script1Values = np.loadtxt(fileNameToLoad)

#splits the above sublists into the 3 lists we used in script1
InputValue, Func1Output, Func2Output=[],[],[]
for i in script1Values:
    InputValue.append(i[0])
    Func1Output.append(i[1])
    Func2Output.append(i[2])



#we  use scipy to fit models parametes
#first for the custFunc1
#Func3
initVals3 = [0.,0.,0.,0.]
best_vals_l, covar_l = opt.curve_fit(custFunc3, InputValue, Func1Output, p0=initVals3)
outputValuesPredicted1=[]
for i in InputValue:
    outputValuesPredicted1.append(custFunc3(i, best_vals_l[0], best_vals_l[1],best_vals_l[2],best_vals_l[3]))

#Func4
initVals4 = [0.,0.,0.,0.,0.]
best_vals_2, covar_2 = opt.curve_fit(custFunc4, InputValue, Func1Output, p0=initVals4)
outputValuesPredicted2=[]
for i in InputValue:
    outputValuesPredicted2.append(custFunc4(i, best_vals_2[0], best_vals_2[1],best_vals_2[2],best_vals_2[3],best_vals_2[4]))

#do a plot using all fits
plt.plot(InputValue, Func1Output, 'r--', label='raw data')
plt.plot(InputValue, outputValuesPredicted1, 'g-', label='3ou')
plt.plot(InputValue, outputValuesPredicted2, 'b.', label='4ou')
plt.legend(loc='best')
plt.title('Function1')
plt.xlabel('InputValue')
plt.ylabel('Func1Output&Predicted')
plt.show()

# και μετά πάνω στην custFunc2
# custFunc2 – πολυώνυμο 3ου βαθμού      custFunc2 – πολυώνυμο 4ου βαθμού
#Func3
initVals3 = [0.,0.,0.,0.]
best_vals_3, covar_3 = opt.curve_fit(custFunc3, InputValue, Func2Output, p0=initVals3)
outputValuesPredicted3=[]
for i in InputValue:
    outputValuesPredicted3.append(custFunc3(i, best_vals_3[0], best_vals_3[1],best_vals_3[2],best_vals_3[3]))

#Func4
initVals4 = [0.,0.,0.,0.,0.]
best_vals_4, covar_4= opt.curve_fit(custFunc4, InputValue, Func2Output, p0=initVals4)
outputValuesPredicted4=[]
for i in InputValue:
    outputValuesPredicted4.append(custFunc4(i, best_vals_4[0], best_vals_4[1],best_vals_4[2],best_vals_4[3],best_vals_4[4]))

#Func8
initVals8 = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
best_vals_8, covar_8= opt.curve_fit(custFunc8, InputValue, Func2Output, p0=initVals8)
outputValuesPredicted8=[]
for i in InputValue:
    outputValuesPredicted8.append(custFunc8(i, best_vals_8[0], best_vals_8[1],best_vals_8[2],best_vals_8[3],best_vals_8[4],best_vals_8[5],best_vals_8[6],best_vals_8[7],best_vals_8[8]))

#FuncC
initValsC = [0,1.,1.]
best_vals_C, covar_C= opt.curve_fit(custFuncC, InputValue, Func2Output, p0=initValsC)
outputValuesPredictedC=[]
for i in InputValue:
    outputValuesPredictedC.append(custFuncC(i, best_vals_C[0], best_vals_C[1],best_vals_C[2]))

#do a plott using all fits
plt.plot(InputValue, Func2Output, 'k-', label='raw data')
plt.plot(InputValue, outputValuesPredicted3, 'm.', label='3ou')
plt.plot(InputValue, outputValuesPredicted4, 'r.', label='4ou')
plt.plot(InputValue, outputValuesPredicted8, 'b.', label='8ou')
plt.plot(InputValue, outputValuesPredictedC, 'y--', label='Cosine')
plt.legend(loc='best')
plt.title('Function2')
plt.xlabel('InputValue')
plt.ylabel('Func2Output&Predicted')
plt.show()


fig = plt.figure(tight_layout=True)
fig.set_tight_layout(True)
ax= fig.add_subplot(211)
ax.bar(['mae1_3','mae1_4' ,'mse1_3' ,'mse1_4'],[mae(Func1Output, outputValuesPredicted1),mae(Func1Output, outputValuesPredicted2),mse(Func1Output, outputValuesPredicted1),mse(Func1Output, outputValuesPredicted2)])
ax.set_xlabel('We observe that 4s are comparatively slightly better than 3s. \nAs a result 4th degree function offers more accuracy')
plt.title('Function1')

ax2=fig.add_subplot(212)
ax2.bar(['mae2_3','mae2_4' ,'mse2_3' ,'mse2_4'],[mae(Func2Output,outputValuesPredicted3),mae(Func2Output,outputValuesPredicted4),mse(Func2Output,outputValuesPredicted3),mse(Func2Output,outputValuesPredicted4)])
ax2.set_xlabel('We observe that 4s are comparatively slightly better than 3s. \nAs a result 4th degree function offers more accuracy\n(Maes are higher than mse due to the square that it has been used)')
plt.title('Function2')
plt.show()
