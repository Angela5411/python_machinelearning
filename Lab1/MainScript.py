#Sfyridaki Angeliki
#cs151036 cs151036@uniwa.gr
#mhxanikwn plhroforikhs

#required packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import traceback
import matplotlib.pyplot as plt
from keras.utils.data_utils import get_file
import cv2
import pandas as pd
import numpy as np
import random
import time
from os import listdir
#import skimage.metrics
import skimage.measure
from pandas.plotting import scatter_matrix

saveList1=[]    #[[type,score,mse,nrmse]]
saveList2=[]    #[[name,noise]]
saveList =[]   #[[name,noise,type,score,mse,nrmse]]


def window_function(title,img,N1,N2):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # this allows for resizing using mouse
    img = np.array(img)
    cv2.imshow(title, img)
    cv2.resizeWindow(title, N1, N2)

def average_function(img,N):
    t = time.time()
    ksize = (N, N)
    blured = cv2.blur(img, ksize)
    tmpRunTime = time.time() - t
    return blured,tmpRunTime

def gaussian_function(img_BGR,N):
    t = time.time()
    blur = cv2.GaussianBlur(img_BGR, (N, N), 0)
    tmpRunTime = time.time() - t
    return blur,tmpRunTime

def median_function(img_BGR,N):
    t = time.time()
    blur = cv2.medianBlur(img_BGR, N)
    tmpRunTime = time.time() - t
    return blur,tmpRunTime

def bilateral_function(img_BGR,N1,N2):
    t = time.time()
    blur = cv2.bilateralFilter(img_BGR, N1, N2, N2)
    tmpRunTime = time.time() - t
    return blur,tmpRunTime

def measure_function(img_GRAY, blur_GRAY, type, tmpRunTime):
    (score, _) = skimage.measure.compare_ssim(img_GRAY, blur_GRAY, full=True)
    mse=skimage.measure.compare_mse(img_GRAY, blur_GRAY)
    nrmse=skimage.measure.compare_nrmse(img_GRAY, blur_GRAY)

    print(type+" filter time was: {:.8f}".format(tmpRunTime), " seconds. ")
    print("SSIM score: {:.4f}".format(score))
    print("MSE score: {:.4f}".format(mse)+'(less is better)')
    print("NRMSE score: {:.4f}".format(nrmse)+'(less is better)')

    saveList1.append([type,'{:.4f}'.format(score),'{:.4f}'.format(mse),'{:.4f}'.format(nrmse)])

#define function to create some noise to an image
def sp_noise(image,prob=0.13):
    '''
    Add salt and pepper noise to image. Replaces random pixels with 0 or 1.
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

#gauss noise Gaussian-distributed additive noise.
def gauss_noise(image,mean=0,var=0.1,sigma=0.5):
    row,col,ch= image.shape
    sigma = var**sigma
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch).astype('uint8')
    noisy = image + gauss
    return noisy

#poisson Poisson-distributed noise generated from the data
def poisson_noise(image,prob=2):
    vals = len(np.unique(image))
    vals = prob ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy

def MainScript_Q1():
    img_BGR = cv2.imread(path + random.choice(list_files))
    img_GRAY = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
    h, w = img_BGR.shape[1], img_BGR.shape[0]

    # and illustrate the results
    window_function("Original Image", img_BGR, h, w)
    # step 2 apply some filters and compare change to the original
    print('Illustrating the effects of various filters, in terms of time and information loss')

    # a. averaging filter-----------------------------------------------
    blur, tmpRunTime = average_function(img_BGR, 5)
    # illustrate the results
    window_function("Averaging blur Filter", blur, h, w)
    blur= cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # calculate the similarity between the images. compute the Structural Similarity Index (SSIM) between the two images
    measure_function(img_GRAY, blur, "Averaging blur", tmpRunTime)

    # b. gaussian filter---------------------------------------------------
    blur, tmpRunTime = gaussian_function(img_BGR, 5)
    # illustrate the results
    window_function("Gauss blur Filter", blur, h, w)
    # calculate the similarity between the images. compute the Structural Similarity Index (SSIM) between the two images
    blur= cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    measure_function(img_GRAY, blur, "Gauss blur", tmpRunTime)

    # c. median filter----------------------------------------------------
    blur, tmpRunTime = median_function(img_BGR, 5)
    # illustrate the results
    window_function("Median Filter", blur, h, w)
    # calculate the similarity between the images. compute the Structural Similarity Index (SSIM) between the two images
    blur= cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    measure_function(img_GRAY, blur, "Median blur", tmpRunTime)

    # d. bilateral filter----------------------------------------------------
    blur, tmpRunTime = bilateral_function(img_BGR, 9, 5)
    # illustrate the results
    window_function("Bilateral Filter", blur, h, w)
    # calculate the similarity between the images. compute the Structural Similarity Index (SSIM) between the two images
    blur= cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    measure_function(img_GRAY, blur, "Bilateral blur", tmpRunTime)

    cv2.waitKey(0)


    fileNameToSave = './OutputFile/Results.txt'  # define the txt file to pass the examples
    headerValues = 'Filter Name | Performance score 1 value | Performance score 2 value | Performance score 3 value'  # column headers
    try:
        np.savetxt(fileNameToSave, saveList1, fmt='%s', header=headerValues)
    except:
        print('error while writing data')
        traceback.print_exc(file=sys.stdout)
    else:
        print('successfully writen')

def noise_function(img_BGR):
    h, w = img_BGR.shape[1], img_BGR.shape[0]
    #img_GRAY = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)

    #create the new noisy image using salt and peper,gauss
    if np.random.randint(2) is 0:
        img_NOISY_BGR = sp_noise(img_BGR, 0.08)
        saveList2[1].append('salt&peper')
    else:
        img_NOISY_BGR=gauss_noise(img_BGR)
        saveList2[1].append('gauss')

    # img_NOISY_BGR=poisson_noise(img_BGR)

    #img_NOISY_GRAY = cv2.cvtColor(img_NOISY_BGR, cv2.COLOR_BGR2GRAY)

    # window_function("Original (noisy) Image",img_NOISY_BGR,h,w)
    score=[]
    # print("Demonstrating the noise reduction capabilities for each of the filters")
    # score=measure_function(img_GRAY,img_NOISY_BGR,"Original (noisy) Image",time.time())
    # cv2.waitKey()
    return img_NOISY_BGR

def MainScript_Q2():
    images=[]
    noisy_images=[] #img_NOISY_BGR
    filtered_images=[]  #[filters: [[blur,"type"],...]]
    for i in list_files:
        img_BGR = cv2.imread(path + i)
        images.append(img_BGR)
        saveList2[0].append(i)
        noisy_images.append(noise_function(img_BGR))
    print(" .. Attempting to fix results, using different filters")
    for i in noisy_images:
        img_BGR=i
        h, w = img_BGR.shape[1], img_BGR.shape[0]
        filters=[]

        # a. averaging filter-----------------------------------------------
        blur, tmpRunTime = average_function(img_BGR, 5)
        # illustrate the results
        window_function("Averaging blur Filter", blur, h, w)
        filters.append([blur,"Averaging blur Filter"])

        # b. gaussian filter---------------------------------------------------
        blur, tmpRunTime = gaussian_function(img_BGR, 5)
        # illustrate the results
        window_function("Gauss blur Filter", blur, h, w)
        filters.append([blur,"Gauss blur Filter"])


        # c. median filter----------------------------------------------------
        blur, tmpRunTime = median_function(img_BGR, 5)
        # illustrate the results
        window_function("Median Filter", blur, h, w)
        filters.append([blur,"Median Filter"])

        # d. bilateral filter----------------------------------------------------
        blur, tmpRunTime = bilateral_function(img_BGR, 9, 5)
        # illustrate the results
        window_function("Bilateral Filter", blur, h, w)
        filters.append([blur,"Bilateral Filter"])

        filtered_images.append(filters)
        cv2.waitKey(0)
    print(np.shape(filtered_images),np.shape(img_BGR))

    for i,j in zip(filtered_images,images):
        img_GRAY = cv2.cvtColor(j, cv2.COLOR_BGR2GRAY)
        for k in i:
            blur = cv2.cvtColor(k[0], cv2.COLOR_BGR2GRAY)
            measure_function(img_GRAY, blur,k[1],time.time())


    j=0
    for i,ii in zip(saveList2[0],saveList2[1]):
        for k in range(4):
            temp=[]
            temp.append(i)
            temp.append('|')
            temp.append(ii)
            temp.append('|')
            temp.append(saveList1[j+k][0])
            temp.append('|')
            temp.append(saveList1[j+k][1])
            temp.append('|')
            temp.append(saveList1[j+k][2])
            temp.append('|')
            temp.append(saveList1[j+k][3])
            saveList.append(temp)
            print(temp)

        j+=4





    # print(saveList)
    #Image ID | Noise type | Filter Name |score 1 | score 2
    fileNameToSave = './OutputFile/Results2.txt'  # define the txt file to pass the examples
    headerValues = 'Image ID| Noise Type| Filter Name | Performance score 1 value | Performance score 2 value | Performance score 3 value'  # column headers
    try:
        np.savetxt(fileNameToSave, saveList, fmt='%s', header=headerValues)
    except:
        print('error while writing data')
        traceback.print_exc(file=sys.stdout)
    else:
        print('successfully writen')


    """MNIST handwritten digits dataset."""
def load_data(path='mnist.npz'):
    """Loads the MNIST dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    # https://s3.amazonaws.com/img-datasets/mnist.npz
    path = get_file(path,
                    origin='C:/Program Files/Python37/Lib/mnist.npz',
                    file_hash='8a61469f7ea1b51cbae51d4f78837e45')
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)


def pick_data(x_train,y_train):

    #shuffle data
    final = list(zip(x_train,y_train))
    random.shuffle(final)

    x, y = zip(*final)
    x_train = np.array(x)
    y_train = np.array(y)

    #pick data
    threes=[]
    fives=[]
    eights=[]
    nines=[]
    for x,y in zip(x_train,y_train):
        if y==3:
            if len(threes) <5:
                threes.append(x)
        elif y==5:
            if len(fives) <5:
                fives.append(x)
        elif y==8:
            if len(eights) <5:
                eights.append(x)
        elif y==9:
            if len(nines)<5:
                nines.append(x)
        if len(threes) is 5 and len(fives)is 5 and len(eights)is 5 and len(nines)is 5:
            break

    return threes,fives,eights,nines

def show_numbers(threes, fives, eights, nines):
    # show data
    plt.subplot(321)
    plt.imshow(threes[0], cmap=plt.get_cmap('gray'))
    plt.subplot(322)
    plt.imshow(threes[1], cmap=plt.get_cmap('gray'))
    plt.subplot(323)
    plt.imshow(threes[2], cmap=plt.get_cmap('gray'))
    plt.subplot(324)
    plt.imshow(threes[3], cmap=plt.get_cmap('gray'))
    plt.subplot(325)
    plt.imshow(threes[4], cmap=plt.get_cmap('gray'))
    plt.title('3')
    plt.show()

    plt.subplot(321)
    plt.imshow(fives[0], cmap=plt.get_cmap('gray'))
    plt.subplot(322)
    plt.imshow(fives[1], cmap=plt.get_cmap('gray'))
    plt.subplot(323)
    plt.imshow(fives[2], cmap=plt.get_cmap('gray'))
    plt.subplot(324)
    plt.imshow(fives[3], cmap=plt.get_cmap('gray'))
    plt.subplot(325)
    plt.imshow(fives[4], cmap=plt.get_cmap('gray'))
    plt.title('5')
    plt.show()

    plt.subplot(321)
    plt.imshow(eights[0], cmap=plt.get_cmap('gray'))
    plt.subplot(322)
    plt.imshow(eights[1], cmap=plt.get_cmap('gray'))
    plt.subplot(323)
    plt.imshow(eights[2], cmap=plt.get_cmap('gray'))
    plt.subplot(324)
    plt.imshow(eights[3], cmap=plt.get_cmap('gray'))
    plt.subplot(325)
    plt.imshow(eights[4], cmap=plt.get_cmap('gray'))
    plt.title('8')
    plt.show()

    plt.subplot(321)
    plt.imshow(nines[0], cmap=plt.get_cmap('gray'))
    plt.subplot(322)
    plt.imshow(nines[1], cmap=plt.get_cmap('gray'))
    plt.subplot(323)
    plt.imshow(nines[2], cmap=plt.get_cmap('gray'))
    plt.subplot(324)
    plt.imshow(nines[3], cmap=plt.get_cmap('gray'))
    plt.subplot(325)
    plt.imshow(nines[4], cmap=plt.get_cmap('gray'))
    plt.title('9')
    plt.show()


def fourier_transform(image):
    # now do the fourier stuff
    f = np.fft.fft2(image)  # find Fourier Transform
    fshift = np.fft.fftshift(f)  # move zero frequency component (DC component) from top left to center
    frequency=fshift.copy()
    # and calculate the magitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    magnitude_spectrum_img = np.round(magnitude_spectrum).astype('uint8')
    #window_function("Magnitude spectrum",magnitude_spectrum_img,480,360)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    lim=int(rows/12)
    fshift[crow - lim:crow + lim+1, ccol - lim:ccol + lim+1] = 0

    # plt.imshow(np.abs(fshift), "gray"), plt.title("Decentralized")
    # plt.show()

    # now you go back to the original image
    f_ishift = np.fft.ifftshift(fshift)  # get DC back to original space
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.round(np.real(img_back)).astype('uint8')  # we need this for the opencv library
    # window_function("Frequency to Spatial",img_back,480,360)

    measure_function(image,img_back,'high pass filter in frequency space',time.time())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return np.abs(frequency),img_back

def show_transforms(frequency):
    fig,ax=plt.subplots(nrows=4, ncols=5)
    rows = ['Row {}'.format(row) for row in ['3', '5', '8', '9']]
    for axes, row in zip(ax[:, 0], rows):
        axes.set_ylabel(row, rotation=0, size='large')

    for row in range(4):
        for col in range(5):
            ax[row,col].imshow(np.abs(frequency[row*5+col]))# ,"gray"
            #print(row*5+col)
    plt.show()

def spatial_frequency_window(spatial,frequency):
    N = int(len(frequency)/4)

    plt.subplot(421)
    plt.imshow(frequency[0], cmap=plt.get_cmap('gray'))
    plt.subplot(422)
    plt.imshow(spatial[0], cmap=plt.get_cmap('gray'))
    plt.subplot(423)
    plt.imshow(frequency[N], cmap=plt.get_cmap('gray'))
    plt.subplot(424)
    plt.imshow(spatial[N], cmap=plt.get_cmap('gray'))
    plt.subplot(425)
    plt.imshow(frequency[2 * N], cmap=plt.get_cmap('gray'))
    plt.subplot(426)
    plt.imshow(spatial[2 * N], cmap=plt.get_cmap('gray'))
    plt.subplot(427)
    plt.imshow(frequency[3 * N], cmap=plt.get_cmap('gray'))
    plt.subplot(428)
    plt.imshow(spatial[3 * N], cmap=plt.get_cmap('gray'))
    plt.show()


def SSIM_similarity_matrix(img):
    result=[]
    for i in range(20):
        k=[]
        for j in range(20):
            k.append([])
        result.append(k)
    #print(np.shape(result))

    for i,ivalue in enumerate(img):
        for j,jvalue in enumerate(img):
            (score, _) = skimage.measure.compare_ssim(ivalue, jvalue ,full=True)
            result[i][j]=score
    print(result)

    fig, ax = plt.subplots()
    ax.imshow(result)
    ax.set_xticks(np.arange(len(img)))
    ax.set_yticks(np.arange(len(img)))
    plt.ylim(-.5, 19.5)
    plt.xlim(-.5, 19.5)


    for i in range(len(img)):
        for j in range(len(img)):
            ax.text(j, i, "%0.1f"%result[i][j],ha="center", va="center", color="w")
    ax.set_title("Similarity Matrix")
    fig.tight_layout()
    plt.show()

    # plt.imshow(result)
    # plt.colorbar()
    # plt.show()


def MainScript_Q3():
    (x_train, y_train), (x_test, y_test) = load_data()

    #Κρατείστε 5 τυχαίες εικόνες για κάθε ένα από τους ακόλουθους αριθμούς: τρία(3), πέντε(5), οκτώ(8) και εννέα(9).
    threes,fives,eights,nines=pick_data(x_train, y_train)
    #show_numbers(threes,fives,eights,nines)
    img=list(threes+fives+eights+nines)

    show_numbers(threes,fives,eights,nines)
    frequency,img_back=[],[]
    for x in img:
        f,b=fourier_transform(x)
        frequency.append(f)
        img_back.append(b)
    show_transforms(frequency)
    spatial_frequency_window(img,frequency)
    SSIM_similarity_matrix(img)
    SSIM_similarity_matrix(frequency)
    # SSIM_similarity_matrix(img_back)

    log,frequency, img_back = [], [], []
    for x in img:
        blur = cv2.GaussianBlur(x, (3, 3), 0)
        l=cv2.Laplacian(blur,cv2.CV_64F)
        f, b = fourier_transform(l)
        log.append(l)
        frequency.append(f)
        img_back.append(b)
    spatial_frequency_window(log,frequency)
    SSIM_similarity_matrix(frequency)
    #SSIM_similarity_matrix(img_back)



####----Main------------------------------------------------------------

#load the image
path = './InputFiles/Images/'
list_files = [f for f in listdir(path)]
#MainScript_Q1()
saveList2.append([])
saveList2.append([])
MainScript_Q2()
#MainScript_Q3()
