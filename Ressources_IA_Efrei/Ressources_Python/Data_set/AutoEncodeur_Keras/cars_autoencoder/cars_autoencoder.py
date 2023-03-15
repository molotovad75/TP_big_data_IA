# -*- coding: utf-8 -*-

#modif. du dossier de travail
import os
os.chdir("D:/DataMining/Databases_for_mining/dataset_for_soft_dev_and_comparison/autoencoder/tutoriel")

#librairie pandas
import pandas

#chargement de la première feuille de données
X = pandas.read_excel("cars_autoencoder.xlsx",header=0,index_col=0,sheet_name=0)

#liste des variables
print(X.info())

#affichage
print(X)

#dimension
print(X.shape)

#nombre d'observations
n = X.shape[0]

#nombre de variables
p = X.shape[1]

#outil centrage réduction
from sklearn.preprocessing import StandardScaler

#instanciation
std = StandardScaler()

#transformation
Z = std.fit_transform(X)
print(Z)

#min pour chaque variable
import numpy
numpy.min(Z,axis=0)

#max. pour chaque variable
numpy.max(Z,axis=0)

#outil couches
from keras.layers import Input, Dense

#outil modélisation
from keras.models import Model

#définition de la structure - couche d'entrée
inputL = Input(shape=(p,))

#couche intermédiaire
hiddenL = Dense(2,activation='sigmoid')(inputL)

#couche de sortie
outputL = Dense(p,activation='linear')(hiddenL)

#modele
autoencoder = Model(inputL,outputL)

#compilation
autoencoder.compile(optimizer='adam',loss='mse')

#affichage des carac. du modèle
print(autoencoder.summary())

#apprentissage à partir des données
n_epochs = 10000
historique = autoencoder.fit(x=Z,y=Z,epochs=n_epochs)

#importation librairie graphique
import matplotlib.pyplot as plt

#affichage de l'évolution de l'apprentissage
plt.plot(numpy.arange(1,n_epochs+1),historique.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Décroissance fnct de perte')
plt.show()

#qualité de l'approximation
print(autoencoder.evaluate(x=Z,y=Z))

#projection des individus dans l'espace initial
ZP = autoencoder.predict(Z)
print(numpy.round(ZP,3))

#que l'on peut calculer manuellement
MSE = 0.0
for i in range(n):
    MSE = MSE + numpy.mean((Z[i,:]-ZP[i,:])**2)
MSE = MSE/n
print(MSE)

#affichage des poids - reconstitution du réseau
print(autoencoder.get_weights())

#modèle "encodeur"
encoder = Model(inputL,hiddenL)

#projection - coordonnées "factorielles"
coord = encoder.predict(Z)
print(coord)

#positionnement des individus dans le plan
fig, axes = plt.subplots(figsize=(15,15))
axes.set_xlim(0.1,0.9)
axes.set_ylim(0.1,0.9)

for i in range(coord.shape[0]):
    plt.annotate(X.index[i],(coord[i,0],coord[i,1]))

plt.title('Position des véhicules dans le plan')
plt.xlabel('Coord 1')
plt.ylabel('Coord 2')
plt.show()

#corrélations des variables avec les "composantes"
correlations = numpy.zeros(shape=(6,2))
for j in range(coord.shape[1]):
    print("Corr. facteur n° %i" % (j))
    for i in range(Z.shape[1]):
        correlations[i,j]=numpy.corrcoef(coord[:,j],Z[:,i])[0,1]
        
#data.frame pour correlations
dfCorr = pandas.DataFrame(correlations,columns=['Corr_1','Corr_2'],index=X.columns)
print(dfCorr)

#cercle des corrélations
fig, axes = plt.subplots(figsize=(10,10))
axes.set_xlim(-1.0,1.0)
axes.set_ylim(-1.0,1.0)
#position des variables
for i in range(dfCorr.shape[0]):
    plt.annotate(dfCorr.index[i],(correlations[i,0],correlations[i,1]))
#axes
plt.plot([0,0],[-1,+1],linestyle='--',c='gray',linewidth=1)
plt.plot([-1,+1],[0,0],linestyle='--',c='gray',linewidth=1)
#cosmétique
circle = plt.Circle((0,0),radius=1,fill=False,edgecolor='gray')
axes.add_artist(circle)
axes.set_aspect(1)
plt.show()

#et déstandardisation de la projection dans l'espace initial
XP = std.inverse_transform(ZP)

#affichage - valeur tronquées pour affichage plus clair
dp = pandas.DataFrame(XP.astype('int'),columns=X.columns,index=X.index)
print(dp)

#calculer les écarts-type des variables initiales
etX = X.apply(lambda x: numpy.std(x))
print(etX)

#signaler les écarts forts et leur sens
gap_et = 0.6
dstr = X.transform(lambda x: x.astype('str'))
for j in range(X.shape[1]):
    for i in range(X.shape[0]):
        if (X.values[i,j]-XP[i,j]) > gap_et * etX[j]:
            dstr.iloc[i,j] = '+%s' % (X.iloc[i,j])
        elif (XP[i,j] - X.values[i,j]) > gap_et * etX[j]:
            dstr.iloc[i,j] = '-%s' % (X.iloc[i,j])
        else:
            dstr.iloc[i,j] = '.'
print(dstr)



#chargement des individus supplémentaires
ind_supp = pandas.read_excel("cars_autoencoder.xlsx",header=0,index_col=0,sheet_name=1)
print(ind_supp)

#transformation
z_ind_supp = std.transform(ind_supp)
print(z_ind_supp)

#projection
coord_ind = encoder.predict(z_ind_supp)
print(coord_ind)


#positionnement des individus dans le plan
fig, axes = plt.subplots(figsize=(15,15))
axes.set_xlim(0.1,0.9)
axes.set_ylim(0.1,0.9)

#individus actifs
for i in range(coord.shape[0]):
    plt.annotate(X.index[i],(coord[i,0],coord[i,1]),c="gray")
    
#individus supplémentaires
for i in range(coord_ind.shape[0]):
    plt.annotate(ind_supp.index[i],(coord_ind[i,0],coord_ind[i,1]),c="blue",fontweight='bold')

plt.title('Individus supplémentaires - Les modèles Peugeot',c='blue')
plt.xlabel('Coord 1')
plt.ylabel('Coord 2')
plt.show()

#variables supplémentaires
var_supp = pandas.read_excel("cars_autoencoder.xlsx",header=0,index_col=0,sheet_name=2)
print(var_supp)

#vérifier que nous avons bien les mêmes individus
print(numpy.sum(X.index != var_supp.index))

#créer un data.frame intermédiaire
dfSupp = pandas.DataFrame(coord,columns=['coord1','coord2'],index=X.index)
dfSupp = pandas.concat([dfSupp,var_supp],axis=1,sort=False)
print(dfSupp)

#corrélations avec prix
corPrix = dfSupp[['coord1','coord2']].corrwith(dfSupp.prix)
print(corPrix)

#corrélations avec co2
corCo2 = dfSupp[['coord1','coord2']].corrwith(dfSupp.co2)
print(corCo2)

#rajouter dans le cercle des corrélations
fig, axes = plt.subplots(figsize=(10,10))
axes.set_xlim(-1.0,1.0)
axes.set_ylim(-1.0,1.0)

#var. actives
for i in range(dfCorr.shape[0]):
    plt.annotate(dfCorr.index[i],(correlations[i,0],correlations[i,1]))
    
#var illustratives
plt.annotate('Prix',(corPrix[0],corPrix[1]),c='blue',fontweight='bold')
plt.annotate('CO2',(corCo2[0],corCo2[1]),c='green',fontweight='bold')
    
plt.plot([0,0],[-1,+1],linestyle='--',c='gray',linewidth=1)
plt.plot([-1,+1],[0,0],linestyle='--',c='gray',linewidth=1)

circle = plt.Circle((0,0),radius=1,fill=False,edgecolor='gray')
axes.add_artist(circle)
axes.set_aspect(1)
plt.show()

#graphique - prix
plt.scatter(data=dfSupp,x='coord1',y='coord2',c='prix',cmap='Oranges')
plt.title('Véhicules illustrés par les prix')
plt.show()


#graphique - co2
plt.scatter(data=dfSupp,x='coord1',y='coord2',c='co2',cmap='Blues')
plt.annotate('MAZDA RX8',(dfSupp.coord1[21]-0.09,dfSupp.coord2[21]+0.025))
plt.title('Véhicules illustrés par le CO2')
plt.show()

#moyennes conditionnelles
dfSupp.pivot_table(index='carburant',values=['coord1','coord2'],aggfunc=pandas.Series.mean)

#catégories de carburant
catCarburant = numpy.unique(dfSupp.carburant)
print(catCarburant)

#graphique -- ilustration selon le type de carburant
fig, ax = plt.subplots()
for cat,col in zip(catCarburant,['blue','green','red']):
    id = numpy.where(dfSupp.carburant==cat)[0]
    ax.scatter(dfSupp.coord1[id],dfSupp.coord2[id],c=col,label=cat)
ax.legend()
plt.title('Véhicules selon le type de carburant')
plt.show()
