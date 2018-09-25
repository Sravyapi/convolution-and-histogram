
# coding: utf-8

# 2D convolution with 3*3 sobel filter

# In[ ]:


#import os
#os.chdir('/Users/sp/Desktop/CVIP/PA1')


# In[2]:


from skimage.exposure import rescale_intensity
from skimage import color
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:



image = cv2.imread('lena_gray.jpg',cv2.IMREAD_GRAYSCALE)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
sobel_y = sobel_x.transpose()

Gx = np.zeros_like(image)
Gy = np.zeros_like(image)

padded_image= np.zeros((image.shape[0] + 2, image.shape[1] + 2))
padded_image[1:-1, 1:-1] = image


# In[4]:



def conv(padded_image,sobel):
    sobelImage = np.zeros_like(image,dtype="float32")
    for i in range(1, padded_image.shape[0]-1):
        for j in range(1, padded_image.shape[1]-1):
            gradient = (sobel[0][0] * padded_image[i-1][j-1]) + (sobel[0][1] * padded_image[i-1][j]) + (sobel[0][2] * padded_image[i-1][j+1]) + (sobel[1][0] * padded_image[i][j-1]) + (sobel[1][1] * padded_image[i][j]) + (sobel[1][2] * padded_image[i][j+1]) + (sobel[2][0] * padded_image[i+1][j-1]) + (sobel[2][1] * padded_image[i+1][j]) + (sobel[2][2] * padded_image[i+1][j+1])
            sobelImage[i-1][j-1] = gradient
    sobelImage = rescale_intensity(sobelImage, in_range=(0, 255))* 255
    return sobelImage


# In[5]:


#Plotting

plt.imshow(image, cmap='gray')
plt.show()

Gx2D=conv(padded_image,sobel_x)
plt.imshow(Gx2D, cmap='gray')
plt.show()

Gy2D=conv(padded_image,sobel_y)
plt.imshow(Gy2D, cmap='gray')
plt.show()

G = np.sqrt(Gx2D**2 + Gy2D**2)
plt.imshow(G, cmap='gray')
plt.show()


# In[6]:


import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from skimage.exposure import rescale_intensity


# In[7]:


image = cv2.imread('lena_gray.jpg', cv2.IMREAD_GRAYSCALE)

sobelx1 = np.array([1,2,1])[:,None] #column
sobelx2 = np.array([-1,0,1])[None,:] #row

sobely1 = sobelx2.transpose() #column
sobely2 = sobelx1.transpose() #row


# In[8]:


#Calculate gx and gy by convolution using first 1D Sobel separable filter
def conv1(image,sobel):
    sobelImage = np.zeros_like(image,dtype="float32")
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            gradient = sobel[0][0] * image[i-1][j] +  sobel[1][0] * image[i][j] + sobel[2][0] * image[i+1][j]
            sobelImage[i-1][j-1] = gradient
    return sobelImage


# In[9]:


sobelImagex=conv1(image,sobelx1)
sobelImagey=conv1(image,sobely1)


# In[10]:



def conv2(sobelImage,sobel):
    padded_image = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    padded_image[1:-1, 1:-1] = sobelImage
    for i in range(1,padded_image.shape[0]-1):
        for j in range(1,padded_image.shape[1]-1):
            gradient = sobel[0][0] * padded_image[i][j-1] + sobel[0][1] * padded_image[i][j] + sobel[0][2] * padded_image[i][j+1]
            sobelImage[i-1][j-1] = gradient
    sobelImage = rescale_intensity(sobelImage, in_range=(0, 255))* 255
    return sobelImage


# In[11]:


#Plotting

Gx1D=conv2(sobelImagex,sobelx2)
Gy1D=conv2(sobelImagey,sobely2)

plt.imshow(image, cmap='gray')
plt.show()

plt.imshow(Gx1D, cmap='gray')
plt.show()

plt.imshow(Gy1D, cmap='gray')
plt.show()




#         1.d                 2D Convolution 101*101 filter

# In[12]:



from skimage.exposure import rescale_intensity
from skimage import color
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
image = cv2.imread('lena_gray.jpg',cv2.IMREAD_GRAYSCALE)


# In[13]:



sobel_x = np.random.rand(101,101)
sobel_y = sobel_x.transpose()

Gx = np.zeros_like(image)
Gy = np.zeros_like(image)

padded_image = np.pad(image, (50,50), 'edge')


# In[14]:


def conv(padded_image,sobel):
    sobelImage = np.zeros_like(image,dtype="float32")
    for y in range(50,padded_image.shape[0] -50):
        for x in range(50,padded_image.shape[1] -50):
            sobelImage[y-50,x-50]=(sobel * padded_image[y-50:y+51,x-50:x+51]).sum()
    sobelImage = rescale_intensity(sobelImage)
    sobelImage = (sobelImage * 255)
    return sobelImage


# In[15]:


plt.imshow(image, cmap='gray')
plt.show()

timestart = time.clock() #Start time
Gx2D=conv(padded_image,sobel_x)
timeend = time.clock() - timestart  #End time
print("2D Convolution with Sobel Filters in x direction: ", timeend)

plt.imshow(Gx2D, cmap='gray')
plt.show()

#Start time
timestart = time.clock()
Gy2D=conv(padded_image,sobel_y)
#End time
timeend = time.clock() - timestart
print("2D Convolution with Sobel Filters in y direction: ", timeend)

plt.imshow(Gy2D, cmap='gray')
plt.show()


#     1.d                 1D Convolution 101*101 filter

# In[16]:


import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from skimage.exposure import rescale_intensity


# In[17]:


image = cv2.imread('lena_gray.jpg', cv2.IMREAD_GRAYSCALE)

sobelx1 = np.random.rand(101,1) #column
sobelx2 = np.random.rand(1,101) #row

#Sobel Separable Filter in y direction
sobely1 = sobelx2.transpose()
sobely2 = sobelx1.transpose()


# In[18]:


#Calculate gx and gy by convolution using first 1D Sobel separable filter
def conv1(image,sobel):
    sobelImage = np.zeros_like(image,dtype="float32")
    for y in range(50,image.shape[0]-50):
        for x in range(50,image.shape[1]-50):
            sobelImage[y-50,x-50]=(sobel * image[y-50:y+51,x-50:x+51]).sum()
    return sobelImage

def conv2(sobelImage,sobel):
    padded_image = np.pad(image, (50,50), 'edge')
    for y in range(50,padded_image.shape[0]-50):
        for x in range(50,padded_image.shape[1]-50):
            sobelImage[y-50,x-50]=(sobel * padded_image[y-50:y+51,x-50:x+51]).sum()
    sobelImage = rescale_intensity(sobelImage)
    return sobelImage


# In[19]:



plt.imshow(image, cmap='gray')
plt.show()

timestart = time.clock()    #Start time
sobelImagex=conv1(image,sobelx1)
Gx1D=conv2(sobelImagex,sobelx2)
timeend = time.clock() - timestart
print("1D Convolution with Sobel Separable Filters in x direction: ", timeend)

plt.imshow(Gx1D, cmap='gray')
plt.show()

#Start time
timestart = time.clock()
sobelImagey=conv1(image,sobely1)
Gy1D=conv2(sobelImagey,sobely2)
timeend = time.clock() - timestart
print("1D Convolution with Sobel Separable Filters in y direction: ", timeend)

plt.imshow(Gy1D, cmap='gray')
plt.show()
