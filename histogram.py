
# coding: utf-8

# In[1]:


#Set the path
#import os
#os.chdir('/Users/sp/Desktop/CVIP/PA1')


# In[2]:


#import libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt


# In[3]:


#read the image unchanged
unchanged = cv2.imread('lena_gray.jpg',cv2.IMREAD_UNCHANGED)
#Read the image in grayscale
grayimage = cv2.imread('lena_gray.jpg',cv2.IMREAD_GRAYSCALE)


# In[4]:


#Calculating Histogram H by scanning every pixel and increment each element of H

def hist(grayimage):
    H = np.zeros(256)
    for row in range(grayimage.shape[0]):
        for col in range(grayimage.shape[1]):
            H[grayimage[row][col]] += 1
    return H

H = hist(grayimage) #histogram

#Calculating Cumulative histogram C

def cumulative(H):
    CH = np.zeros(256)
    CH[0] = H[0]
    for row in range(1, 256):
        CH[row] = CH[row-1] + H[row]
    return CH

CH = cumulative(H) #cumulative histogram

#Creating look up table

def grayLevels(CHp):
    Ti = round(float(256 - 1)/(grayimage.shape[0] * grayimage.shape[1]) * CHp)
    return Ti

def FillLookUp(CH):
    T = np.zeros(256)
    for i in range(256):
        T[i] = grayLevels(CH[i])
    return T

T = FillLookUp(CH) #Transformation function

#Calculating enhanced image by mapping pixel intensities of unchanged image into enhanced image

def enhanced(T):
    EnhancedImage = np.zeros_like(grayimage)
    for row in range(grayimage.shape[0]):
        for col in range(grayimage.shape[1]):
            EnhancedImage[row][col] = T[grayimage[row][col]]
    return EnhancedImage

EnhancedImage = enhanced(T)

#Equalizing the histogram

def Equalized(EnhancedImage):
    EqualizedHistogram = np.zeros(256)
    for row in range(grayimage.shape[0]):
        for col in range(0, grayimage.shape[1]):
            EqualizedHistogram[EnhancedImage[row][col]] += 1
    return EqualizedHistogram

EqualizedHistogram = Equalized(EnhancedImage)


# In[6]:


#Plotting image, grayscale image, enhanced image (with better contrast),
#Histogram, Cumulative histogram, Transformation function,Equalized Histogram

plt.imshow(unchanged)
plt.xticks([]), plt.yticks([])
plt.title('Original Image')
plt.show()

plt.imshow(grayimage,cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('Grayscale Image')
plt.show()

plt.imshow(EnhancedImage,cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('Enhanced Image')
plt.show()

plt.plot(H)
plt.xlabel('Intensity')
plt.ylabel('No. of Pixels')
plt.title('Histogram of the image')
plt.show()

plt.plot(CH)
plt.xlabel('Intensity')
plt.ylabel('No. of Pixels')
plt.title('Cumulative Histogram')
plt.show()

plt.plot(T)
plt.xlabel('Original intensity')
plt.ylabel('New intensity')
plt.title('Transformation Function')
plt.show()

plt.plot(EqualizedHistogram)
plt.xlabel('Intensity')
plt.ylabel('No. of Pixels')
plt.title('Equalized Histogram')
plt.show()
