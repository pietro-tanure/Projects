import numpy as np
import cv2
import sys

from sklearn.cluster import KMeans

if len(sys.argv) != 2:
  print ("Usage :",sys.argv[0],"<Image_in>")
  sys.exit(2)
else:
  img_bgr=cv2.imread(sys.argv[1],-1)
(h_img,w_img,c) = img_bgr.shape
print("Dimension de l'image :",h_img,"lignes x",w_img,"colonnes x",c,"canaux")
print("Type de l'image :",img_bgr.dtype)

# Création des clusters (entraînement) sur une image
Nb_classes = 2
img_samples = np.reshape(img_bgr,(-1,3))
print(img_samples.shape)
kmeans = KMeans(n_clusters=Nb_classes, random_state=0).fit(img_samples)
# Affichage des centres de cluster 
print("Centres des clusters : ",kmeans.cluster_centers_)
# Affichage des labels dans l'image d'entraînement
print('kmeans.labels', kmeans.labels_.shape)
img_labels = np.reshape(kmeans.labels_,(h_img,w_img))
print("Type de l'image de label :",img_labels.dtype)
# Normalisation pour affichage
img_labels_display = (img_labels*255)/(Nb_classes - 1)
img_labels_display = img_labels_display.astype(np.uint8)
cv2.imshow("Clusters dans l'image (train)",img_labels_display)
cv2.waitKey(0)

# Pour tester sur une autre image : 
img_bgr=cv2.imread('Images_Classif/Essex_Faces/94/ajsega.19.jpg',-1)
(h_img,w_img,c) = img_bgr.shape
img_samples = np.reshape(img_bgr,(-1,3))
img_labels =kmeans.predict(img_samples)
print(img_labels.shape)
img_labels = np.reshape(img_labels,(h_img,w_img))

img_labels_display = (img_labels*255)/(Nb_classes - 1)
img_labels_display = img_labels_display.astype(np.uint8)
cv2.imshow("Clusters dans l'image (train)",img_labels_display)
cv2.waitKey(0)
