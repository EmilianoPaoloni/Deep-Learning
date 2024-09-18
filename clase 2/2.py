import RN_Perceptron as rn
import os
import pandas as pd
import numpy as np
path="/Users/Neno/Documents/Facultad/Deep Learning/Practica2"
os.chdir(path)
data = pd.read_csv('hojas.csv')

###########NUMERIZACION################
# atributo sexo con codificación binaria
NuevasColumnas = pd.get_dummies(data['Clase'])

# Agregamos las nuevas columnas al DataFrame
data = pd.concat([data,NuevasColumnas ], axis=1)

# Borramos la columna original y 1 de las nuevas
data.drop(['Hoja'],axis=1, inplace=True)

T=np.array(data.iloc[:,3])
X=np.array(data.iloc[:,:2])
dibuja=0
titulos=['Atrib1','atrib2']
#[w,b,ite]=rn.entrena_Perceptron(X,T,0.01,300,dibuja,titulos)
#//Y=[]
suma_iteraciones=0
suma_aciertos=0
for i in range(50):
    [w,b,ite]=rn.entrena_Perceptron(X,T,0.01,500,dibuja,titulos)
    Y=rn.aplica_Perceptron(X,w,b)
    nAciertos=sum(Y==T)
    suma_aciertos=suma_aciertos+nAciertos
    suma_iteraciones=suma_iteraciones+ite
    print ("caso " + str(i+1) + ": %% de aciertos=%.2f %%" %(100*nAciertos/X.shape[0]))
print("cantidad promedio de iteraciones= "+str(suma_iteraciones/50))
print("cantidad promedio de aciertos=%.2f %% "%(100*suma_aciertos/50/X.shape[0]))
    