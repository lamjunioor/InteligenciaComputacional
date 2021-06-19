import numpy as np
import pandas as pd

x=[]
x = np.loadtxt('dados.txt') #lê os dados e coloca em uma matriz

w=[]
for i in range(4):
  w.append(round(np.random.uniform(0.0, 1.0),2)) #inicializa o vetor de pesos com valores aleatorios
n=0.01 #inicializa a taxa de aprendizado
print("Pesos iniciais: ",w)

epoca=0
while True:
  erro = False
  for j in range(x.T[0].size):
    u=0
    for k in range(len(w)):
      u+= x[j][k]*w[k]
    if u>=0:
      y=1.0
    else:
      y=-1.0
    for k in range(len(w)):
      if (x[j][4] != y):
        w[k]=w[k]+n*(x[j][4]-y)*x[j][k]
        erro = True
  epoca+=1  
  if erro==False:
    break

print("Pesos finais: ", w)
print("Perceptron treinado em:",epoca," épocas")

teste=[]
teste = np.loadtxt('teste.txt') #lê os dados e coloca em uma matriz

resultado=[]

for j in range(teste.T[0].size):
  u=0
  for k in range(len(w)):
    u+= teste[j][k]*w[k]
  if u>=0:
    y=1.0
  else:
    y=-1.0
  resultado.append(y)

print("Resultado final: ",resultado)