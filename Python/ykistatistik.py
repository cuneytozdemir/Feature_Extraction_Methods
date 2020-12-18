# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 17:21:57 2019

@author: aidata
"""
import numpy as np
import math

def mean(lst):
    return round(sum(lst) / len(lst),2)
def std(lst):
    m = mean(lst)
    variance = sum([(value - m) ** 2 for value in lst])
    return round(math.sqrt(variance / len(lst)),2)
def Energy(lst):
    return round(math.sqrt(sum(i**2 for i in lst)/len(lst)),2)
def ShannonsEntropy(dizi):
    hist = {}; l = 0;
    for e in dizi:
        l += 1
        if e not in hist:
            hist[e] = 0
        hist[e] += 1
        
    elist = []
    
    for v in hist.values():
        elist.append(-(v / l * math.log(v / l ,2))) # c ve log karesi alınan drumlar var.
        
    return round(sum(elist),2)
def Sequentialabsolutedifferences(lst):
    return round(sum(abs(lst[i+1]-lst[i]) for i in range(len(lst)-1))/len(lst),2)
def Kurtosis(lst):
   # topla=0
   # for i in lst:
   #  topla += ((i-np.mean(lst))**4)/len(lst) 
   # return round(topla / (np.std(lst)**4),2)
    ort=mean(lst)
    return round(sum([((i-ort) **4)/len(lst)  for i in lst]) / (std(lst)**4),2)
def Skewness(lst):
    ort=mean(lst)
    return round(sum([((i-ort) **3)/len(lst)  for i in lst]) / (std(lst)**3),2)
def median(lst):
    return round(np.median(np.array(lst)),2)
def maximum(lst):
    return round(np.max(lst),2)
def minumum(lst):
    return round(np.min(lst),2)
def CoefficientofVariance(lst):
    return round(np.mean(lst)/np.std(lst),2)

def FeatureFormula(dizi):
    entropi=np.array([])
    entropi=np.append(entropi,mean(dizi))
    entropi=np.append(entropi,std(dizi))
    entropi=np.append(entropi,Energy(dizi))
    entropi=np.append(entropi,ShannonsEntropy(dizi))
    entropi=np.append(entropi,Sequentialabsolutedifferences(dizi))
    entropi=np.append(entropi,Kurtosis(dizi))
    entropi=np.append(entropi,Skewness(dizi))
    entropi=np.append(entropi,median(dizi))
    entropi=np.append(entropi,maximum(dizi))
    entropi=np.append(entropi,minumum(dizi))
    entropi=np.append(entropi,CoefficientofVariance(dizi))
    return entropi