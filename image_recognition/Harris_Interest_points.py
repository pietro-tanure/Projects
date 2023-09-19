import numpy as np
import cv2
from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")
print("Type de l'image :",img.dtype)

#Début du calcul
t1 = cv2.getTickCount()
Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)

#Calcul de la fonction d'intérêt de Harris
#Calcul de Ix
kernel_dx = np.array([[-1,0,1]])
img_dx = cv2.filter2D(img,-1,kernel_dx)

#Calcul de Iy
kernel_dy = np.array([[-1],[0],[1]])
img_dy = cv2.filter2D(img,-1,kernel_dy)

#Matrice d'autocorrelation
kernel_voisinage = np.array(np.ones((11,11))) #voisinage 7x7
X11 = cv2.filter2D(img_dx**2,-1,kernel_voisinage)
X12 = cv2.filter2D(img_dx*img_dy,-1,kernel_voisinage)
X22 = cv2.filter2D(img_dy**2,-1,kernel_voisinage)

#Theta
Theta = (X11*X22-X12**2) - 0.06* (X11+X22) ** 2

# Calcul des maxima locaux et seuillage
Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.01
se = np.ones((d_maxloc,d_maxloc),np.uint8)
Theta_dil = cv2.dilate(Theta,se)

#Suppression des non-maxima-locaux
Theta_maxloc[Theta < Theta_dil] = 0.0

#On néglige également les valeurs trop faibles
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Mon calcul des points de Harris :",time,"s")
print("Nombre de cycles par pixel :",(t2 - t1)/(h*w),"cpp")


plt.subplot(131)
plt.imshow(img,cmap = 'gray')
plt.title('Image originale')

plt.subplot(132)
plt.imshow(Theta,cmap = 'gray')
plt.title('Fonction de Harris')

se_croix = np.uint8([[1, 0, 0, 0, 1],
[0, 1, 0, 1, 0],[0, 0, 1, 0, 0],
[0, 1, 0, 1, 0],[1, 0, 0, 0, 1]])
Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
#Relecture image pour affichage couleur
Img_pts=cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
(h,w,c) = Img_pts.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes x",c,"canaux")
print("Type de l'image :",Img_pts.dtype)
print("Nombre de points de Harris:", np.count_nonzero(Theta_ml_dil > 0))
#On affiche les points (croix) en rouge
Img_pts[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(133)
plt.imshow(Img_pts)
plt.title('Points de Harris')

plt.show()
