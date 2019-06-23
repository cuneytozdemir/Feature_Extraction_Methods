# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:23:40 2019

@author: Cüneyt ÖZDEMİR cuneytozdemir33@gmail.com
"""
import MLClassification as mlc
import NewAttributeExtraction as yk
import ykistatistik as yki
import pandas as pd
import numpy as np
import math,os
from sklearn import preprocessing 
from email.parser import Parser

def readmail(folder):
    body=""
    try:
        parser = Parser()
        with open(folder, 'r', encoding='ISO-8859-1') as ifile:
                email = parser.parse(ifile)       
        if email.is_multipart():
            for part in email.walk():
                content_type = part.get_content_type()
                content_dispo = str(part.get('Content-Disposition'))    
                # skip any text/plain (txt) attachments
                if content_type == 'text/plain' and 'attachment' not in content_dispo:
                    body = part.get_payload(decode=False)
                    break ## only keep the first email
        else:
            body = email.get_payload(decode=False)
    except: 
        pass
    return body 


yk.PencereBoyutu=4
yk.alfa=1
yk.beta=3
yk.komsuluk=4*yk.beta
yk.MA="Y"  #M=motif, A=Açı, V=vektör,Y=İSTATİSTİK


if (yk.MA=="M"):
    Results=np.zeros(1*(math.factorial(yk.PencereBoyutu)+1))[np.newaxis] #pencere boyutu ayarlanıyor
elif (yk.MA=="V"):
    Results=np.zeros(257)[np.newaxis]
elif (yk.MA=="A"):
    Results=np.zeros(361)[np.newaxis]
else :Results=np.zeros(12)[np.newaxis]

#for say in range(2):
#    if (say==0):
#        folder = r"D:\Dropbox\Doktora\veriseti\duyguanalizi\imdb\train\neg"            
#        spham=0            
#    else : 
#        folder = r"D:\Dropbox\Doktora\veriseti\duyguanalizi\imdb\train\pos"
#        spham=1
#    print(folder)
#        
#    for dosya in os.listdir(folder) :
#        uza=os.path.join(folder,dosya)             
#        body=readmail(uza)
#        if (body!=""):
#                try:
#                    df=np.array([])   
#                    add=np.array([]) 
#                    df=np.array([np.append(df,x) for x in map(ord,yk.rootsoftheliturgical(body))])     
#                    add=np.append(add,yk.aciozellik(df))  
#                    add=np.append(add,spham) #sınıflandırma için
#                    Results=np.concatenate([Results,add[np.newaxis, :]])   
#                    say +=1
#                except:
#                    say +=1
#Results = np.delete(Results, (0), axis=0)
df = pd.read_csv(r'D:\Dropbox\Doktora\python\MachineLearnging\leapfm.cvs',encoding='ISO-8859-1')
df = pd.DataFrame(df).fillna(method='ffill')
x_train = df.iloc[:, :-1].values
y_train = df.iloc[: ,-1].values

#sk=preprocessing.LabelEncoder()
#y_train=sk.fit_transform(y_train)

say=0
for row in x_train:  
    try:
        df=np.array([])   
        add=np.array([]) 
      #  df=np.array([np.append(df,x) for x in map(ord,yk.rootsoftheliturgical(row))])     
        add=np.append(add,yki.FeatureFormula(row))  
        add=np.append(add,y_train[say]) #sınıflandırma için
        Results=np.concatenate([Results,add[np.newaxis, :]])   
        say +=1
    except:
        say +=1
Results = np.delete(Results, (0), axis=0)
sonuclar=mlc.classification(pd.DataFrame(Results))
display(sonuclar.sort_values('Accuracy_Score', ascending=False))
