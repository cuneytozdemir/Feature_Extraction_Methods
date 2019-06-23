# -*- coding: utf-8 -*-
# Meta informations.
__version__ = '1.1.1'
__author__ = 'Cüneyt ÖZDEMİR'
__author_email__ = 'cuneytozdemir33@gmail.com'


################################################################################
# Yılmaz Kaya && Cüneyt ÖZDEMİR
################################################################################

import numpy as np
import pandas as pd
import re,math,itertools
from string import punctuation
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk
from snowballstemmer import stemmer


MA="M"
komsuluk=4
beta=1
alfa=1
PencereBoyutu=4

def StopWords(words):
   # words=htmlextract(words)
    stop_words = set(stopwords.words('turkish'))               
    word_tokens = word_tokenize(words)            
    return [w for w in word_tokens if not w in stop_words] 

def htmlextract(sentence):
    cleanr = re.compile('<.*?>')
    sentence = re.sub(r' ', '', sentence) 
    sentence = re.sub(cleanr, ' ', sentence)        #Removing HTML tags
    sentence = re.sub(r'[?|!|\'|"|#|@]',r'',sentence)
    sentence = re.sub(r'[.|,|-|_|;|`|)|(|\|]',r' ',sentence)        #Removing Punctuations
    sentence=sentence.replace(' ','')
    return sentence    

def rootsoftheliturgical(words):
    words=words.lower()
    rootfind = stemmer('turkish')
    trans=str.maketrans('', '', punctuation)
    words = words.translate(trans)
    words=StopWords(words)
   # letters = words.split()
    letters = rootfind.stemWords(words)
    string =' '.join(letters)
    return string

def indisBul(D):
    S=np.sort(D)
    ind=np.column_stack((D, np.zeros(np.shape(D)[0])))
    k=0
    kontrol=0
    while (k<(len(D))) :
    #for k in range(len(D)-1):
        bul=np.where(D==S[k])
        if (len(bul[0])) >1 and (kontrol==0):
            ind[bul[0][0]][1]=k+1
            ind[bul[0][1]][1]=k+2
            kontrol=-1
            k=k+1
        elif len(bul[0])==1 :
            ind[bul[0][0]][1]=k+1
            kontrol=0            
        k=k+1
    return ind                 
def motifyeni(veri):
    P=np.array(list(itertools.permutations(range(1,PencereBoyutu+1))))# pencere boyuna bağlı tüm permütasyonlar
    Pc=np.zeros(1*len(P))
    indx=0  
    veri = np.delete(veri, np.in1d(veri, 32).nonzero()[0])
    veri = np.delete(veri, np.in1d(veri, 253).nonzero()[0])
    veri = np.delete(veri, np.in1d(veri, 254).nonzero()[0]) 
    for i in range(len(veri)):
        a=veri[i:i+PencereBoyutu]   # pencere boyutu kadar sinyalden alınan parça          
        b=indisBul(a)[:,-1]; # alınan parçanın örüntüsünün elde edilmesi
        try:
            patindx= pd.DataFrame(np.sum(abs(P-np.matlib.repmat(b,len(P),1)),axis=1)) #hangi patterne uyduğu bulunuyor    
            patindx=pd.DataFrame(patindx.values[::-1], patindx.index, patindx.columns) # Ters çevirdi

            indx = patindx[patindx[0]==0].index.values.astype(int)[0]
            Pc[indx]=Pc[indx]+1; # bulunan patter sayacı 1 arttırılıyor
                
        except: 
            indx=1   
    return Pc;
def aci(veri):
    acilar=[]
    for i in range(len(veri)-2) :
        P=veri[i:i+3]
        z=np.array([[1],[2],[3]])
        z=np.append(z,P,axis=1)
        dd=np.linalg.det([(z[1,:]-z[0,:]),(z[2,:]-z[1,:])])
        dtt=np.dot(z[1,:]-z[0,:],z[2,:]-z[1,:])        
        ang = math.atan2(abs(dd),dtt);
        ac=round(ang*180/math.pi)+180
        acilar=np.append(acilar, ac)    
    return acilar
def vectmap(veri):
  #  count=np.zeros(256)  
    try:
        Lists=np.zeros(8)[np.newaxis] 
        l=len(veri)
        veri=veri.reshape((veri.shape[0])) 
        for i in range(l):
            binlist=np.array([])
            kon=np.ones(2*komsuluk//beta)
            kon.fill(veri[i])
            
            dat=np.zeros(2*komsuluk//beta)
            if (i<=komsuluk+alfa-2):
                if (i>alfa-1):
                    dat[math.ceil((komsuluk-i+alfa-1)//beta):komsuluk//beta:1]=veri[0:i-alfa+1:beta]
                dat[(komsuluk//beta):2*komsuluk//beta]=veri[(i+alfa-1):(i+komsuluk+alfa-1):beta]
            elif ((i+komsuluk+alfa)>l):
                dat[0:komsuluk//beta:1]=veri[(i-komsuluk-alfa+1):(i-alfa+1):beta] 
                if ((i+alfa-1)<=l) :
                    dat[(komsuluk//beta):math.ceil((komsuluk+l-i-alfa+1)/beta)]=veri[(i+alfa-1):l:beta]
            elif ((i>komsuluk+alfa-2) and (i<=(l-komsuluk-alfa))) :
                dat[0:komsuluk//beta]=veri[(i-komsuluk-alfa+1):(i-alfa+1):beta] # burada problem var gibi beta 2 olunca
                dat[(komsuluk//beta):2*komsuluk//beta]=veri[(i+alfa-1):(i+komsuluk+alfa-1):beta]
        
            binlist=np.append(binlist,kon<dat)   # küçükse 0 büyükse 1 olmalı 
            Lists=np.concatenate([Lists,binlist[np.newaxis, :]])
    except:
        print("hata oluştu")
    Lists = np.delete(Lists, (0), axis=0)
    b=np.packbits(Lists[:,::-1].astype(np.uint8),axis=-1)
    return b
def aciozellik(veri):
    if (MA=="M"):
        return motifyeni(veri)  
    elif (MA=="A"):
        a=aci(veri)
        say=360
    else: 
        a=vectmap(veri)
        say=256    
		https://github.com/cuneytozdemir/MotifOruntuler.py/blob/master/Motif%C3%96r%C3%BCnt%C3%BC.py
    
    count=np.zeros(say)
    for k in range (say) :
        ss=len(np.where(a==(k))[0])
        if ss >0 :
            count[k]=ss
    return count 
