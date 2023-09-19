import numpy as np
import cv2

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png',0))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")

#Méthode directe
t1 = cv2.getTickCount()
img1 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)

#print image originale
#plt.subplot(121)
#plt.imshow(img,cmap = 'gray')
#plt.title('Image originale')


for y in range(1,h-1):
  for x in range(1,w-1):
    val_orig = 5 * img[y, x] - img[y - 1, x] - img[y, x - 1] - img[y + 1, x] - img[y, x + 1]
    img1[y,x] = min(max(val_orig,0),255)


t2 = cv2.getTickCount()
time = (t2 - t1) / cv2.getTickFrequency()
print("Méthode directe :",time,"s")

#Dérivés et norme
h_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
h_y = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
img4 = cv2.filter2D(img,-1,h_x)
img5 = cv2.filter2D(img,-1,h_y)
img2 = ((img4**2)+(img5**2))**(1/2)

#print images

plt.subplot(141)
plt.imshow(img1,cmap = 'gray')
plt.title('Convolution \n - Méthode Directe')

plt.subplot(142)
plt.imshow(img2,cmap = 'gray')
plt.title('Convolution \n - Module du gradient')

plt.subplot(143)
plt.imshow(img4,cmap = 'gray')
plt.title('Convolution \n - Dérivé en x')

#plt.subplot(144)
#plt.imshow(img5,cmap = 'gray')
#plt.title('Convolution - Dérivé en y')


#Méthode filter2D  (même chose qu'avant mais avec une fonction de l'OpenCV)
t1 = cv2.getTickCount()
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
img3 = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1) / cv2.getTickFrequency()
print("Méthode filter2D :",time,"s")

plt.subplot(144)
plt.imshow(img3,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Convolution - filter2D')

plt.show()
