# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:25:09 2022

@author: 
    Colin Duchanoy
    Hugo Quin
    Hugo Lancery
    Vincent Soucourre
"""
#%%Importation des différentes bibliothèques
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kernels import *
import cv2 
import matplotlib.pyplot as plt
import mabiblio as mbl
from tqdm import tqdm

#%% 
"""
-------------------Exercice 1 : Reconstruction de surface avec les moindres carrées à noyaux-----------------
""" 
#%%

P=np.array([[0,0,1],
            [1,0,1.5],
            [0,1,0.5],
            [1,1,1],
            [0.5,0.5,0]])

#faisons une liste des kernels importé
#Il faut exécuter le programme kernel pour pouvoir importer les différents kernels
Liste_kernels=[kern_gauss,kern_sigmo,kern_ratio_quad,kern_mquad,kern_inv_mquad,kern_cauchy,kern_log]


mbl.Visualisation(P,mbl.MCsurface(P)[0],nb=5)

#Calcul du kernel optimal
b=[] #Liste qui contiendra les différentes erreurs des kernels
for i in range (len(Liste_kernels)):
    b.append(mbl.MCKsurface(P,Liste_kernels[i])[1])
j=b.index(min(b)) #comparaison du meilleur kernel

mbl.VisualisationK(P,mbl.MCKsurface(P,Liste_kernels[j])[0],nb=20, kern=Liste_kernels[j])    
print("Erreur optimale MCKsurface avec noyau: ", mbl.MCKsurface(P,Liste_kernels[j])[1])


#%% 
"""
-------------------Exercice 2 : Partitionnement d’image avec ACP et k-moyennes-------------------------------
"""
#%% 
"""
-----------Partie 1 : L’ACP le retour du retour du retour...-----------
"""
#%% Question 4

#Test de la robustesse, en utilisant des données déjà rencontrées

#%% Données Iris
Xbrut=np.genfromtxt("iris.csv" , dtype=str , delimiter=',') #données brutes
labelscolonne=Xbrut[ 0 , : -1]
labelsligne=Xbrut[1 : , -1]
#%%
X=Xbrut[1 : , : -1].astype('float')
#%%


#%% Données Mnist
train_data=np.loadtxt('mnist_train.csv', delimiter=',') 
labelsligne=train_data[: , 0]
#%%
X=train_data[ : , 1 : ].astype('float')
#%%

#%% 
q=mbl.Kaiser(X)
Xq=mbl.ACP(X,q)

#%% Question 5
# On crée la matrice diagonale W et on la modifie selon les critères que l'on veut valoriser
n,d=np.shape(X)
W=np.eye(d)
W[0,0]=400 #Ici on va mettre en valeur la ligne du pixel
Xqp=mbl.ACPond(X,q,W)


#%% 
"""
-----------Partie 2 : L’algorithme des k-moyennes-----------
""" 
#%% Test de l'algorithme 

S,Sind=mbl.Kmoy(X,3,10**(-15),True)


#%%
"""
-----------Partie 3 : L’algorithme de partitionnement-----------
"""

#Choix de l'image 
#%%
image=cv2.imread("zebre.jpg")
#%%
image=cv2.imread("feuille.jpg")
#%%
image=cv2.imread("batiment.jpg")
#%%
image=cv2.imread("lena.jpg")
#%%
image=cv2.imread("montagne.jpg")
#%%
image=cv2.imread("neb.jpg")
#%%
image=cv2.imread("paysage.jpg")
#%%
image=cv2.imread("immeuble.jpg")
#%%

#Version Non-Pondérée

nbpts=25 #nombre de points random par ligne
n,m,p=np.shape(image)

qkayser=2 #On force qkayser à 2 car notre fonction kaiser ne fonctionne pas

data_img=mbl.data_pixels(image, mbl.choixpts(image, nbpts)[1],nbpts)

S,Sind = mbl.Kmoyimg(data_img,qkayser,0)  

imgmasqueponctuel=mbl.Masque(Sind,image,nbpts,data_img)    
    
cv2.imwrite("resultat1.jpg", imgmasqueponctuel)     

imgmasque=mbl.RemplissageMasque(imgmasqueponctuel.astype('uint8') * 255)

cv2.imwrite("resultat2.jpg",imgmasque) 

imgmasquetot1,imgmasquetot2=mbl.Masquetot(image,imgmasque)  

cv2.imwrite("resultat3.jpg", imgmasquetot1)

cv2.imwrite("resultat4.jpg", imgmasquetot2)

#%% 

#Version Pondérée

image=cv2.imread("feuille.jpg")
nbpts=25 #nombre de points random par ligne
n,m,p=np.shape(image)

k=2
data_img=mbl.data_pixels(image, mbl.choixpts(image, nbpts)[1],nbpts)
w=np.eye(np.shape(data_img)[1])

#Possibilité de mettre dans une fonction 

i=0 #0 pour accentuer les lignes
#i=1 #1 pour accentuer les colonnes
#i=2 #2 pour accentuer les rouges
#i=3 #3 pour accentuer les verts
#i=4 #4 pour accentuer les bleus
#i=5 #5 pour accentuer la moyenne des intensités de rouges sur les 9 pixels autours
#i=6 #6 pour accentuer la moyenne des intensités de verts sur les 9 pixels autours
#i=7 #7 pour accentuer la moyenne des intensités de bleus sur les 9 pixels autours
w[i,i]=200

S,Sind = mbl.Kmoyimg(data_img,qkayser,1,w)  

imgmasqueponctuel=mbl.Masque(Sind,image,nbpts,data_img)    
    
cv2.imwrite("resultatpond1.jpg", imgmasqueponctuel)     

imgmasque=mbl.RemplissageMasque(imgmasqueponctuel.astype('uint8') * 255)

cv2.imwrite("resultatpond2.jpg",imgmasque) 

imgmasquetot1,imgmasquetot2=mbl.Masquetot(image,imgmasque)  

cv2.imwrite("resultatpond3.jpg", imgmasquetot1)

cv2.imwrite("resultatpond4.jpg", imgmasquetot2)
