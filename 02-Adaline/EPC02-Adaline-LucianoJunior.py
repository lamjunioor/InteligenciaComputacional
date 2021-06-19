# -*- coding: utf-8 -*-
"""EPC02.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dRFiPnUJwFxQoHl_dlM8hkappdzVIeDO
"""

import numpy as np
import matplotlib.pyplot as plt

x=[]
x = np.loadtxt('dados.txt') #lê os dados e coloca em uma matriz

w=[]
for i in range(5):
  w.append(round(np.random.uniform(0.0, 1.0),3)) #inicializa o vetor de pesos com valores aleatorios
n=0.0025 #inicializa a taxa de aprendizado
print("Pesos iniciais: ",w)
e= 10**-6 #inicializa a taxa de precisão

def eqm():
  aux_eqm=0
  for j in range (x.T[0].size):
    aux_u=0
    for k in range(len(w)):
      aux_u+=x[j][k]*w[k]
    aux_eqm+=(x[j][5]-aux_u)**2
  return aux_eqm/x.T[0].size

epoca=0
eqm_ant=99999
eqm_atual=1
eqm_vetor=[]
while abs(eqm_atual-eqm_ant)>e:
  eqm_ant=eqm_atual;
  for j in range(x.T[0].size):
    u=0
    for k in range(len(w)):
      u+=x[j][k]*w[k]
    for k in range(len(w)):
      w[k]+=n*(x[j][5]-u)*x[j][k]
  epoca+=1
  eqm_atual=eqm()
  eqm_vetor.append(eqm_atual)
print("Pesos finais: ", w)
print("Adaline treinada em: ", epoca, " épocas")

teste=[]
teste = np.loadtxt('teste.txt')

resultado=[]

for j in range (teste.T[0].size):
  u=0
  for k in range (len(w)):
    u+=teste[j][k]*w[k]
  if u>=0:
    y=1.0
  else:
    y=-1.0
  resultado.append(y)
print("Resultado: ", resultado)

plt.plot(eqm_vetor)
plt.ylabel("EQM")
plt.xlabel("Época")
plt.show()