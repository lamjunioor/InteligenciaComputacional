# -*- coding: utf-8 -*-
"""EPC06.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1L9bxU2JJfXLd2sew50pMb5nWfDktUY9P
"""

#!pip install -U scikit-fuzzy

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl####

##Função de ler dados##
def ler_data(dataset):
  arquivo= open(dataset)
  dados=[]
  aux_x=[]
  aux_y=[]
  for linha in arquivo.readlines():
    dados.append(linha.split(', '))

  for i in range(9, len(dados)):
    aux_x.append([dados[i][0], dados[i][1], dados[i][2], dados[i][3]])
    aux_y.append(converter(dados[i][4]))

  aux_x=np.array(aux_x)
  aux_y=np.array(aux_y)

  return (aux_x,aux_y)

###FUNÇÃO CONVERTER###
def converter(y):
  if "Iris-setosa\n"==y:
    return 1
  elif "Iris-versicolor\n"==y:
    return 2
  elif "Iris-virginica\n"==y:
    return 3

def definirRegioes(x_treino):
  min_sl = float(min(x_treino.T[0]))
  max_sl = float(max(x_treino.T[0]))
  sepal_length = np.linspace(min_sl, max_sl, 500)

  sl_p = fuzz.trimf(sepal_length, [min_sl, min_sl, min_sl + (max_sl - min_sl)/2])
  sl_m = fuzz.trimf(sepal_length, [min_sl + (max_sl - min_sl)/4, min_sl + (max_sl - min_sl)/2, min_sl + 3*(max_sl - min_sl)/4])
  sl_g = fuzz.trimf(sepal_length, [min_sl + (max_sl - min_sl)/2, max_sl, max_sl])

  min_sw = float(min(x_treino.T[1]))
  max_sw = float(max(x_treino.T[1]))
  sepal_width = np.linspace(min_sw, max_sw, 500)

  sw_p = fuzz.trimf(sepal_width, [min_sw, min_sw, min_sw + (max_sw - min_sw)/2])
  sw_m = fuzz.trimf(sepal_width, [min_sw + (max_sw - min_sw)/4, min_sw + (max_sw - min_sw)/2, min_sw + 3*(max_sw - min_sw)/4])
  sw_g = fuzz.trimf(sepal_width, [min_sw + (max_sw - min_sw)/2, max_sw, max_sw])

  min_pl = float(min(x_treino.T[2]))
  max_pl = float(max(x_treino.T[2]))
  petal_length = np.linspace(min_pl, max_pl, 500)

  pl_p = fuzz.trimf(petal_length, [min_pl, min_pl, min_pl + (max_pl - min_pl)/2])
  pl_m = fuzz.trimf(petal_length, [min_pl + (max_pl - min_pl)/4, min_pl + (max_pl - min_pl)/2, min_pl + 3*(max_pl - min_pl)/4])
  pl_g = fuzz.trimf(petal_length, [min_pl + (max_pl - min_pl)/2, max_pl, max_pl])

  min_pw = float(min(x_treino.T[3]))
  max_pw = float(max(x_treino.T[3]))
  petal_width = np.linspace(min_pw, max_pw, 500)

  pw_p = fuzz.trimf(petal_width, [min_pw, min_pw, min_pw + (max_pw - min_pw)/2])
  pw_m = fuzz.trimf(petal_width, [min_pw + (max_pw - min_pw)/4, min_pw + (max_pw - min_pw)/2, min_pw + 3*(max_pw - min_pw)/4])
  pw_g = fuzz.trimf(petal_width, [min_pw + (max_pw - min_pw)/2, max_pw, max_pw])
  
  return sepal_length, sl_p, sl_m, sl_g, sepal_width, sw_p, sw_m, sw_g, petal_length, pl_p, pl_m, pl_g, petal_width, pw_p, pw_m, pw_g

def gerar_matriz(l, c):
  matriz = [];
  for i in range(l):
    linha = [];
    for j in range(c):
      linha.append(0);
    matriz.append(linha);
  return matriz;

def gerar_lista(l):
  lista = [];
  for i in range(l):
    lista.append(0);
  return lista;

def wang_mendel(x_treino, y_treino, sepal_length, sl_p, sl_m, sl_g, sepal_width, sw_p, sw_m, sw_g, petal_length, pl_p, pl_m, pl_g, petal_width, pw_p, pw_m, pw_g):
  regras = gerar_matriz(len(x_treino),0)

  for i in range(len(x_treino)):
    sl_pequena = fuzz.interp_membership(sepal_length, sl_p, x_treino[i][0])
    sl_media = fuzz.interp_membership(sepal_length, sl_m, x_treino[i][0])
    sl_grande = fuzz.interp_membership(sepal_length, sl_g, x_treino[i][0])
    
    membership_sl = [sl_pequena, sl_media, sl_grande]
    
    sw_pequena = fuzz.interp_membership(sepal_width, sw_p, x_treino[i][1])
    sw_media = fuzz.interp_membership(sepal_width, sw_m, x_treino[i][1])
    sw_grande = fuzz.interp_membership(sepal_width, sw_g, x_treino[i][1])

    membership_sw = [sw_pequena, sw_media, sw_grande]
    
    pl_pequena = fuzz.interp_membership(petal_length, pl_p, x_treino[i][2])
    pl_media = fuzz.interp_membership(petal_length, pl_m, x_treino[i][2])
    pl_grande = fuzz.interp_membership(petal_length, pl_g, x_treino[i][2])
    
    membership_pl = [pl_pequena, pl_media, pl_grande]
    
    pw_pequena = fuzz.interp_membership(petal_width, sl_p, x_treino[i][3])
    pw_media = fuzz.interp_membership(petal_width, sl_m, x_treino[i][3])
    pw_grande = fuzz.interp_membership(petal_width, sl_g, x_treino[i][3])

    membership_pw = [pw_pequena, pw_media, pw_grande]

    for m in range(len(membership_sl)):
      if (membership_sl[m] == max(membership_sl)) and len(regras[i]) == 0:
        regras[i].append({'region': m + 1 , 'interp_membership': membership_sl[m]})
    
    for m in range(len(membership_sw)):
      if (membership_sw[m] == max(membership_sw)) and len(regras[i]) == 1:
        regras[i].append({'region': m + 1, 'interp_membership': membership_sw[m]})

    for m in range(len(membership_pl)):
      if (membership_pl[m] == max(membership_pl)) and len(regras[i]) == 2:
        regras[i].append({'region': m + 1, 'interp_membership': membership_pl[m]})
    
    for m in range(len(membership_pw)):
      if (membership_pw[m] == max(membership_pw)) and len(regras[i]) == 3:
        regras[i].append({'region': m + 1, 'interp_membership': membership_pw[m]})
    regras[i].append({'classe': y_treino[i]})
    
  rpeso = ishibuchi(x_treino, y_treino, sepal_length, sl_p, sl_m, sl_g, sepal_width, sw_p, sw_m, sw_g, petal_length, pl_p, pl_m, pl_g, petal_width, pw_p, pw_m, pw_g, regras)
  
  rclass = classificarRegras(regras)
  rclasspeso = classificarRegrasPeso(rpeso)
  
  rfinal = removerConflitantes(regras, rclass)
  regras_classe1, regras_classe2, regras_classe3, final_regras = filtrar(rfinal)
  
  rfinalPeso = removerConflitantes(rpeso, rclasspeso)
  regras_classe1Peso, regras_classe2Peso, regras_classe3Peso, final_regrasPeso = filtrar(rfinalPeso)

  return regras, regras_classe1, regras_classe2, regras_classe3, final_regras, rpeso, regras_classe1Peso, regras_classe2Peso, regras_classe3Peso, final_regrasPeso
  

def ishibuchi(x_treino, y_treino, sepal_length, sl_p, sl_m, sl_g, sepal_width, sw_p, sw_m, sw_g, petal_length, pl_p, pl_m, pl_g, petal_width, pw_p, pw_m, pw_g, regras):
  rpeso = regras.copy()
  for r in range(len(rpeso)):
    beta_g1 = 0
    beta_g2 = 0
    beta_g3 = 0
    betas = gerar_lista(3)
    
    for amostra in range(len(x_treino)):
      membership_sl, membership_sw, membership_pl, membership_pw = pertinencia(x_treino[amostra], sepal_length, sl_p, sl_m, sl_g, sepal_width, sw_p, sw_m, sw_g, petal_length, pl_p, pl_m, pl_g, petal_width, pw_p, pw_m, pw_g)
      
      if rpeso[r][4]['classe'] == 1:
        beta_g1 += membership_sl[f"{rpeso[r][0]['region']}"] * membership_sw[f"{rpeso[r][1]['region']}"] * membership_pl[f"{rpeso[r][2]['region']}"] * membership_pw[f"{rpeso[r][3]['region']}"]
      elif rpeso[r][4]['classe'] == 2:
        beta_g2 += membership_sl[f"{rpeso[r][0]['region']}"] * membership_sw[f"{rpeso[r][1]['region']}"] * membership_pl[f"{rpeso[r][2]['region']}"] * membership_pw[f"{rpeso[r][3]['region']}"]
      elif rpeso[r][4]['classe'] == 3:
        beta_g3 += membership_sl[f"{rpeso[r][0]['region']}"] * membership_sw[f"{rpeso[r][1]['region']}"] * membership_pl[f"{rpeso[r][2]['region']}"] * membership_pw[f"{rpeso[r][3]['region']}"]
    
    betas[0] = beta_g1
    betas[1] = beta_g2
    betas[2] = beta_g3
  
    rpeso[r].append({'degree': degree(betas)})
  return rpeso

def degree(betas):
  b = (sum(betas) - max(betas))/(len(betas)-1)
  return abs(max(betas) - b)/sum(betas)

def pertinencia(amostra, sepal_length, sl_p, sl_m, sl_g, sepal_width, sw_p, sw_m, sw_g, petal_length, pl_p, pl_m, pl_g, petal_width, pw_p, pw_m, pw_g):
  sl_pequena = fuzz.interp_membership(sepal_length, sl_p, amostra[0])
  sl_media = fuzz.interp_membership(sepal_length, sl_m, amostra[0])
  sl_grande = fuzz.interp_membership(sepal_length, sl_g, amostra[0])
  
  membership_sl = {'1': sl_pequena, '2': sl_media, '3': sl_grande}
  
  sw_pequena = fuzz.interp_membership(sepal_width, sw_p, amostra[1])
  sw_media = fuzz.interp_membership(sepal_width, sw_m, amostra[1])
  sw_grande = fuzz.interp_membership(sepal_width, sw_g, amostra[1])
  
  membership_sw = {'1': sw_pequena, '2': sw_media, '3': sw_grande}
  
  pl_pequena = fuzz.interp_membership(petal_length, pl_p, amostra[2])
  pl_media = fuzz.interp_membership(petal_length, pl_m, amostra[2])
  pl_grande = fuzz.interp_membership(petal_length, pl_g, amostra[2])
  
  membership_pl = {'1': pl_pequena, '2': pl_media, '3': pl_grande}
  
  pw_pequena = fuzz.interp_membership(petal_width, sl_p, amostra[3])
  pw_media = fuzz.interp_membership(petal_width, sl_m, amostra[3])
  pw_grande = fuzz.interp_membership(petal_width, sl_g, amostra[3])

  membership_pw = {'1': pw_pequena, '2': pw_media, '3': pw_grande}

  return membership_sl, membership_sw, membership_pl, membership_pw

def classificarRegras(regras):
  rclass = []
  for i in range(len(regras)):
    rclass.append(regras[i][0]['interp_membership']*regras[i][1]['interp_membership']*regras[i][2]['interp_membership']*regras[i][3]['interp_membership'])
  return rclass

def classificarRegrasPeso(rpeso):
  rclasspeso = []
  for i in range(len(rpeso)):
    rclasspeso.append(rpeso[i][0]['interp_membership']*rpeso[i][1]['interp_membership']*rpeso[i][2]['interp_membership']*rpeso[i][3]['interp_membership']*rpeso[i][5]['degree'])
  return rclasspeso

def removerConflitantes(regras, score):
  min_membership = set()
  rfinal = []

  for i in range(len(regras) - 1):
    for j in range(i + 1, len(regras)):
      if (regras[i][0]['region'] == regras[j][0]['region']) and (regras[i][1]['region'] == regras[j][1]['region']) and (regras[i][2]['region'] == regras[j][2]['region']) and (regras[i][3]['region'] == regras[j][3]['region']) and (regras[i][4]['classe'] != regras[j][4]['classe']):
        if (score[i] > score[j]):
          min_membership.add(j)
        else:
          min_membership.add(i)

  for i in range(len(regras)):
    if not (i in min_membership):
      rfinal.append(regras[i])

  return rfinal

def filtrar(regras_):
  regras = regras_.copy()
  classe1 = set()
  classe2 = set()
  classe3 = set()  
  
  for i in range(len(regras)):
    if regras[i][4]['classe'] == 1:
      classe1.add(f"{regras[i][0]['region']}{regras[i][1]['region']}{regras[i][2]['region']}{regras[i][3]['region']}")
    elif regras[i][4]['classe'] == 2:
      classe2.add(f"{regras[i][0]['region']}{regras[i][1]['region']}{regras[i][2]['region']}{regras[i][3]['region']}")
    elif regras[i][4]['classe'] == 3:
      classe3.add(f"{regras[i][0]['region']}{regras[i][1]['region']}{regras[i][2]['region']}{regras[i][3]['region']}")
  
  regras_classe1 = []
  regras_classe2 = []
  regras_classe3 = []
  final_regras = []

  for el in classe1:
    text_rule = ''
    rule = []
    for i in range(len(el)):
      if i == 0:
        if el[i] == '1':
          text_rule = "Se o comprimento da sépala é pequena & "
          rule.append({'region': 1})
        elif el[i] == '2':
          text_rule = f"Se o comprimento da sépala é média & "
          rule.append({'region': 2})
        elif el[i] == '3':
          text_rule = "Se o comprimento da sépala é grande & "
          rule.append({'region': 3})
      elif i == 1:
        if el[i] == '1':
          text_rule += "Largura da sépala é pequena & "
          rule.append({'region': 1})
        elif el[i] == '2':
          text_rule += "Largura da sépala é média & "
          rule.append({'region': 2})
        elif el[i] == '3':
          text_rule += "Largura da sépala é grande & "
          rule.append({'region': 3})
      elif i == 2:
        if el[i] == '1':
          text_rule += "Comprimento da pétala é pequena & "
          rule.append({'region': 1})
        elif el[i] == '2':
          text_rule += "Comprimento da pétala é média & "
          rule.append({'region': 2})
        elif el[i] == '3':
          text_rule += "Comprimento da pétala é grande & "
          rule.append({'region': 3})
      elif i == 3:
        if el[i] == '1':
          text_rule += "Largura da pétala é pequena, "
          rule.append({'region': 1})
        elif el[i] == '2':
          text_rule += "Largura da pétala é média, "
          rule.append({'region': 2})
        elif el[i] == '3':
          text_rule += "Largura da pétala é grande, "
          rule.append({'region': 3})
    rule.append({'classe': 1})
    regras_classe1.append(rule)
    text_rule += "Então a planta é Iris-setosa"
    final_regras.append(text_rule)

  for el in classe2:
    text_rule = ''
    rule = []
    for i in range(len(el)):
      if i == 0:
        if el[i] == '1':
          text_rule = "Se o comprimento da sépala é pequena & "
          rule.append({'region': 1})
        elif el[i] == '2':
          text_rule = f"Se o comprimento da sépala é média & "
          rule.append({'region': 2})
        elif el[i] == '3':
          text_rule = "Se o comprimento da sépala é grande & "
          rule.append({'region': 3})
      elif i == 1:
        if el[i] == '1':
          text_rule += "Largura da sépala é pequena & "
          rule.append({'region': 1})
        elif el[i] == '2':
          text_rule += "Largura da sépala é média & "
          rule.append({'region': 2})
        elif el[i] == '3':
          text_rule += "Largura da sépala é grande & "
          rule.append({'region': 3})
      elif i == 2:
        if el[i] == '1':
          text_rule += "Comprimento da pétala é pequena & "
          rule.append({'region': 1})
        elif el[i] == '2':
          text_rule += "Comprimento da pétala é média & "
          rule.append({'region': 2})
        elif el[i] == '3':
          text_rule += "Comprimento da pétala é grande & "
          rule.append({'region': 3})
      elif i == 3:
        if el[i] == '1':
          text_rule += "Largura da pétala é pequena, "
          rule.append({'region': 1})
        elif el[i] == '2':
          text_rule += "Largura da pétala é média, "
          rule.append({'region': 2})
        elif el[i] == '3':
          text_rule += "Largura da pétala é grande, "
          rule.append({'region': 3})
    rule.append({'classe': 2})
    regras_classe2.append(rule)
    text_rule += "Então a planta é Iris-versicolor"
    final_regras.append(text_rule)

  for el in classe3:
    text_rule = ''
    rule = []
    for i in range(len(el)):
      if i == 0:
        if el[i] == '1':
          text_rule = "Se o comprimento da sépala é pequena & "
          rule.append({'region': 1})
        elif el[i] == '2':
          text_rule = f"Se o comprimento da sépala é média & "
          rule.append({'region': 2})
        elif el[i] == '3':
          text_rule = "Se o comprimento da sépala é grande & "
          rule.append({'region': 3})
      elif i == 1:
        if el[i] == '1':
          text_rule += "Largura da sépala é pequena & "
          rule.append({'region': 1})
        elif el[i] == '2':
          text_rule += "Largura da sépala é média & "
          rule.append({'region': 2})
        elif el[i] == '3':
          text_rule += "Largura da sépala é grande & "
          rule.append({'region': 3})
      elif i == 2:
        if el[i] == '1':
          text_rule += "Comprimento da pétala é pequena & "
          rule.append({'region': 1})
        elif el[i] == '2':
          text_rule += "Comprimento da pétala é média & "
          rule.append({'region': 2})
        elif el[i] == '3':
          text_rule += "Comprimento da pétala é grande & "
          rule.append({'region': 3})
      elif i == 3:
        if el[i] == '1':
          text_rule += "Largura da pétala é pequena, "
          rule.append({'region': 1})
        elif el[i] == '2':
          text_rule += "Largura da pétala é média, "
          rule.append({'region': 2})
        elif el[i] == '3':
          text_rule += "Largura da pétala é grande, "
          rule.append({'region': 3})
    rule.append({'classe': 3})
    regras_classe3.append(rule)
    text_rule += "Então a planta é Iris-virginica"
    final_regras.append(text_rule)

  return regras_classe1, regras_classe2, regras_classe3, final_regras

def listaSemZero(lista):
  aux = []
  for l in lista:
    if l > 0:
      aux.append(l)
  return aux
  
def classificar(x_test, y_test, sepal_length, sl_sm, sl_me, sl_bi, sepal_width, sw_sm, sw_me, sw_bi, petal_length, pl_sm, pl_me, pl_bi, petal_width, pw_sm, pw_me, pw_bi, rules1, rules2, rules3):
  classes_mrfc = gerar_lista(y_test.shape[0])
  classes_mrfg = gerar_lista(y_test.shape[0])

  rules_class1 = rules1.copy()
  rules_class2 = rules2.copy()
  rules_class3 = rules3.copy()
  
  t_norm1 = gerar_lista(len(rules_class1))
  t_norm2 = gerar_lista(len(rules_class2))
  t_norm3 = gerar_lista(len(rules_class3))
  
  for sample in range(x_test.shape[0]):
    membership_sl, membership_sw, membership_pl, membership_pw = pertinencia(x_test[sample], sepal_length, sl_sm, sl_me, sl_bi, sepal_width, sw_sm, sw_me, sw_bi, petal_length, pl_sm, pl_me, pl_bi, petal_width, pw_sm, pw_me, pw_bi)

    # Extraindo as mínimas pertinências da regra
    for i in range(len(rules_class1)):
      norm = []
      norm.append(membership_sl[f"{rules_class1[i][0]['region']}"])
      norm.append(membership_sw[f"{rules_class1[i][1]['region']}"])
      norm.append(membership_pl[f"{rules_class1[i][2]['region']}"])
      norm.append(membership_pw[f"{rules_class1[i][3]['region']}"])
      t_norm1[i] = min(norm)

    for i in range(len(rules_class2)):
      norm = []
      norm.append(membership_sl[f"{rules_class2[i][0]['region']}"])
      norm.append(membership_sw[f"{rules_class2[i][1]['region']}"])
      norm.append(membership_pl[f"{rules_class2[i][2]['region']}"])
      norm.append(membership_pw[f"{rules_class2[i][3]['region']}"])
      t_norm2[i] = min(norm)

    for i in range(len(rules_class3)):
      norm = []
      norm.append(membership_sl[f"{rules_class3[i][0]['region']}"])
      norm.append(membership_sw[f"{rules_class3[i][1]['region']}"])
      norm.append(membership_pl[f"{rules_class3[i][2]['region']}"])
      norm.append(membership_pw[f"{rules_class3[i][3]['region']}"])
      t_norm3[i] = min(norm)
   
    # Eliminando as pertinências de valor zeros das mínimas pertinências da regra
    n1 = listaSemZero(t_norm1)
    n2 = listaSemZero(t_norm2)
    n3 = listaSemZero(t_norm3)

    # Classficação utilizando MRFC
    if len(n1) > 0 and len(n2) > 0 and len(n3) > 0:
      if max(n1) > max(n2) and max(n1) > max(n3):
        classes_mrfc[sample] = 1
      if max(n2) > max(n1) and max(n2) > max(n3):
        classes_mrfc[sample] = 2
      if max(n3) > max(n1) and max(n3) > max(n2):
        classes_mrfc[sample] = 3
    elif len(n1) > 0 and len(n2) > 0:
      if max(n1) > max(n2):
        classes_mrfc[sample] = 1
      elif max(n1) < max(n2):
        classes_mrfc[sample] = 2
    elif len(n1) > 0 and len(n3) > 0:
      if max(n1) > max(n3):
        classes_mrfc[sample] = 1
      elif max(n1) < max(n3):
        classes_mrfc[sample] = 3
    elif len(n2) > 0 and len(n3) > 0:
      if max(n2) > max(n3):
        classes_mrfc[sample] = 2
      elif max(n2) < max(n3):
        classes_mrfc[sample] = 3
    elif len(n1) > 0:
      classes_mrfc[sample] = 1
    elif len(n2) > 0:
      classes_mrfc[sample] = 2
    elif len(n3) > 0:
      classes_mrfc[sample] = 3

    # Classficação utilizando MRFG
    if len(n1) > 0 and len(n2) > 0 and len(n3) > 0:
      if np.mean(n1) > np.mean(n2) and np.mean(n1) > np.mean(n3):
        classes_mrfg[sample] = 1
      if np.mean(n2) > np.mean(n1) and np.mean(n2) > np.mean(n3):
        classes_mrfg[sample] = 2
      if np.mean(n3) > np.mean(n1) and np.mean(n3) > np.mean(n2):
        classes_mrfg[sample] = 3
    elif len(n1) > 0 and len(n2) > 0:
      if np.mean(n1) > np.mean(n2):
        classes_mrfg[sample] = 1
      elif np.mean(n1) < np.mean(n2):
        classes_mrfg[sample] = 2
    elif len(n1) > 0 and len(n3) > 0:
      if np.mean(n1) > np.mean(n3):
        classes_mrfg[sample] = 1
      elif np.mean(n1) < np.mean(n3):
        classes_mrfg[sample] = 3
    elif len(n2) > 0 and len(n3) > 0:
      if np.mean(n2) > np.mean(n3):
        classes_mrfg[sample] = 2
      elif np.mean(n2) < np.mean(n3):
        classes_mrfg[sample] = 3
    elif len(n1) > 0:
      classes_mrfg[sample] = 1
    elif len(n2) > 0:
      classes_mrfg[sample] = 2
    elif len(n3) > 0:
      classes_mrfg[sample] = 3

  # Cálculo de acurácia da classificação utilizando MRFC      
  accuracy_mrfc = 0
  for c in range(y_test.shape[0]):
    if classes_mrfc[c] == y_test[c]:
      accuracy_mrfc = accuracy_mrfc + 1

  accuracy_mrfc = (accuracy_mrfc / y_test.shape[0]) * 100

  # Cálculo de acurácia da classificação utilizando MRFG
  accuracy_mrfg = 0
  for c in range(y_test.shape[0]):
    if classes_mrfg[c] == y_test[c]:
      accuracy_mrfg = accuracy_mrfg + 1

  accuracy_mrfg = (accuracy_mrfg / y_test.shape[0]) * 100

  return classes_mrfc, accuracy_mrfc, classes_mrfg, accuracy_mrfg

acc_mrfc = []
acc_mrfg = []
acc_mrfc_peso = []
acc_mrfg_peso = []

for i in range(10):
  ##Ler dados de treino
  x_treino, y_treino= ler_data('iris-10-'+str(i+1)+'tra.dat')

  ##Ler dados de teste
  x_teste, y_teste= ler_data('iris-10-'+str(i+1)+'tst.dat')
  
  # Gerando universo e as regiões fuzzy para cada variável da amostra
  sepal_length, sl_p, sl_m, sl_g, sepal_width, sw_p, sw_m, sw_g, petal_length, pl_p, pl_m, pl_g, petal_width, pw_p, pw_m, pw_g = definirRegioes(x_treino)

  # Aplicando os algoritmos para alcançar as regras finais e as que classificam cada classe
  regras, regras_classe1, regras_classe2, regras_classe3, final_regras, rpeso, regras_classe1Peso, regras_classe2Peso, regras_classe3Peso, final_regrasPeso = wang_mendel(x_treino, y_treino, sepal_length, sl_p, sl_m, sl_g, sepal_width, sw_p, sw_m, sw_g, petal_length, pl_p, pl_m, pl_g, petal_width, pw_p, pw_m, pw_g)
  
  # Classificação utilizando as regras sem peso
  classes_mrfc, acuracia_mrfc, classes_mrfg, acuracia_mrfg = classificar(x_teste, y_teste, sepal_length, sl_p, sl_m, sl_g, sepal_width, sw_p, sw_m, sw_g, petal_length, pl_p, pl_m, pl_g, petal_width, pw_p, pw_m, pw_g, regras_classe1, regras_classe2, regras_classe3)
  
  # Classificação utilizando as regras com peso
  classes_mrfcPeso, acuracia_mrfcPeso, classes_mrfgPeso, acuracia_mrfgPeso = classificar(x_teste, y_teste, sepal_length, sl_p, sl_m, sl_g, sepal_width, sw_p, sw_m, sw_g, petal_length, pl_p, pl_m, pl_g, petal_width, pw_p, pw_m, pw_g, regras_classe1Peso, regras_classe2Peso, regras_classe3Peso)
  
  acc_mrfc.append(acuracia_mrfc)
  acc_mrfg.append(acuracia_mrfg)

  acc_mrfc_peso.append(acuracia_mrfcPeso)
  acc_mrfg_peso.append(acuracia_mrfgPeso)

print('MRFC:')
acc_mrfc = np.array(acc_mrfc)
print('Acurácia: ',acc_mrfc.round(2))
print('Média: ',np.mean(acc_mrfc).round(2))
print('Desvio Padrão: ',np.std(acc_mrfc).round(2))
print('\n')

print('MRFC com peso:')
acc_mrfc_peso = np.array(acc_mrfc_peso)
print('Acurácia: ',acc_mrfc_peso.round(2))
print('Média: ',np.mean(acc_mrfc_peso).round(2))
print('Desvio Padrão: ',np.std(acc_mrfc_peso).round(2))
print('\n')

print('MRFG:')
acc_mrfg = np.array(acc_mrfg)
print('Acurácia: ',acc_mrfg.round(2))
print('Média: ',np.mean(acc_mrfg).round(2))
print('Desvio Padrão: ',np.std(acc_mrfg).round(2))
print('\n')

print('MRFG com peso:')
acc_mrfg_peso = np.array(acc_mrfg_peso)
print('Acurácia: ', acc_mrfg_peso.round(2))
print('Média: ',np.mean(acc_mrfg_peso).round(2))
print('Desvio Padrão: ',np.std(acc_mrfg_peso).round(2))