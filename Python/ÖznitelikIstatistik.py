# -*- coding: utf-8 -*-
# Meta informations.
__version__ = '1.1.1'
__author__ = 'Cüneyt ÖZDEMİR'
__author_email__ = 'cuneytozdemir33@gmail.com'


################################################################################
# Data structures.
################################################################################

import numpy as np
import pandas as pd
import os 
import re
import math
import itertools
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
from string import punctuation
from email.parser import Parser
from sklearn.model_selection import cross_val_predict
from snowballstemmer import stemmer
import numpy.matlib as npm
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import keras
from keras.layers import LSTM
from keras.layers import Flatten

kokbul1 = stemmer('turkish')    

def DataReduction(veri,PBoyut):
    yeni=np.array([np.mean(veri[i*PBoyut:(i+1)*PBoyut]) for i in range(round(len(veri)/PBoyut) )])  
    return yeni
def stopkelime(oku):
    stop_words = set(stopwords.words('english'))               
    word_tokens = word_tokenize(oku)            
    return [w for w in word_tokens if not w in stop_words] 
def htmlayikla(sentence):
    cleanr = re.compile('<.*?>')
    sentence = re.sub(r' ', '', sentence) 
    sentence = re.sub(cleanr, ' ', sentence)        #Removing HTML tags
    sentence = re.sub(r'[?|!|\'|"|#|@]',r'',sentence)
    sentence = re.sub(r'[.|,|-|_|;|`|)|(|\|]',r' ',sentence)        #Removing Punctuations
    sentence=sentence.replace(' ','')
    return sentence    
def koklerineayir(metin):
    cevirici = str.maketrans('', '', punctuation)
    metin = metin.translate(cevirici)
    metin=metin.lower()
    kelimeler = metin.split()
    kelimeler = kokbul1.stemWords(kelimeler)
    string =' '.join(kelimeler)
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
            patindx= pd.DataFrame(np.sum(abs(P-npm.repmat(b,len(P),1)),axis=1)) #hangi patterne uyduğu bulunuyor    
            patindx=pd.DataFrame(patindx.values[::-1], patindx.index, patindx.columns) # Ters çevirdi

            indx = patindx[patindx[0]==0].index.values.astype(int)[0]
            Pc[indx]=Pc[indx]+1; # bulunan patter sayacı 1 arttırılıyor
                
        except: 
            indx=1   
    return Pc;

def motifinsert(MA,mesaj,classification):
    Entropy=np.array([])  
    df=np.array([])
    df=np.array([np.append(df,x) for x in map(ord,htmlayikla(mesaj))])
    if (MA=="M"):
        Entropy=np.append(Entropy,motifyeni(df))
    else:Entropy=np.append(Entropy,aciozellik(MA,df))    
    
    Entropy=np.append(Entropy,classification) #sınıflandırma için   
    return Entropy

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

def aciyeni(veri,L,R):
    acilar=[]
    for i in range(1+L,len(veri)-R) :
        P=veri[i-L:i+R]
        dd=np.linalg.det([(P[0,:]-P[1,:]),(P[1,:]-P[2,:])])
        dtt=np.dot(P[0,:]-P[1,:],P[1,:]-P[2,:])        
        ang = math.atan2(dd,dtt);
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
                if (i+1>alfa):
                    dat[math.ceil((komsuluk-i+alfa-1)//beta):komsuluk//beta]=veri[0:i-alfa+1:beta]
                dat[(komsuluk//beta):2*komsuluk//beta]=veri[(i+alfa):(i+komsuluk+alfa):beta] # ok
            elif ((i+komsuluk+alfa)>l):
                dat[0:komsuluk//beta]=veri[(i-komsuluk-alfa+beta):(i-alfa+1):beta] 
                if (i+alfa<=l) :
                    dat[(komsuluk//beta):math.ceil((komsuluk+l-i-alfa)/beta)]=veri[(i+alfa):l:beta]
            elif ((i>komsuluk+alfa-2) and (i<=(l-komsuluk-alfa))) :
                dat[0:komsuluk//beta]=veri[(i-komsuluk-alfa+beta):(i-alfa+1):beta] # burada problem var gibi beta 2 olunca
                dat[(komsuluk//beta):2*komsuluk//beta]=veri[(i+alfa):(i+komsuluk+alfa):beta]
        
            binlist=np.append(binlist,kon<dat)   # küçükse 0 büyükse 1 olmalı 
            Lists=np.concatenate([Lists,binlist[np.newaxis, :]])
    except:
        print("hata oluştu")
    Lists = np.delete(Lists, (0), axis=0)
    b=np.packbits(Lists[:,::-1].astype(np.uint8),axis=-1)
    return b

def aciozellik(ma,veri):
    if (ma=="A"):
        a=aci(veri,L,R)
        say=360
    if (ma=="AY"):
        a=aciyeni(veri)
        say=360
    elif (ma=="M"):
        motifyeni(df)
    else: 
        a=vectmap(veri)
        say=256    
    
    count=np.zeros(say)
    for k in range (say) :
        ss=len(np.where(a==(k))[0])
        if ss >0 :
            count[k]=ss
    return count 

    
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
def TrecRead_email():
    if (MA=="M"):
        Entropiler=np.zeros(1*(math.factorial(PencereBoyutu)+1))[np.newaxis] #pencere boyutu ayarlanıyor
    else :Entropiler=np.zeros(361)[np.newaxis]  
    folder = "C:/Users/aidata/Desktop/SpamDataset/trec06p/trec06p"
    print(folder)
    foll="/full/Index"
    dosy=open(folder+foll,encoding='ISO-8859-1')
    data=dosy.read()
    datas=data.split('\n')
    
    for k in range(len(datas)):
        try :
            spham=datas[k][:4]
            if (spham=="spam"):
                spham=0
            else: spham=1
            
            dos=datas[k]
            dosyauzantisi=dos[dos.find('../')+2:None]
            gelen=readmail(folder+dosyauzantisi)       
            Entropiler=np.concatenate([Entropiler,motifinsert(gelen,spham)[np.newaxis, :]])
        except:pass
    Entropiler = np.delete(Entropiler, (0), axis=0)
    return Entropiler
def LingSpamRead():
    if (MA=="M"):
        Entropiler=np.zeros(1*(math.factorial(PencereBoyutu)+1))[np.newaxis] #pencere boyutu ayarlanıyor
    elif (MA=="V"):
        Entropiler=np.zeros(257)[np.newaxis]
    else :Entropiler=np.zeros(361)[np.newaxis] 
    folder = r"D:\test"
    print(folder)
    for dosya in os.listdir(folder) :
        uza=os.path.join(folder,dosya) 
        spham=dosya[:5]
        if (spham=="spmsg"):
            spham=0
        else: spham=1
        
        body=readmail(uza)       
        Entropiler=np.concatenate([Entropiler,motifinsert(body,spham)[np.newaxis, :]])
    Entropiler = np.delete(Entropiler, (0), axis=0)
    return Entropiler
def csdm2010SpamRead():
    if (MA=="M"):
        Entropiler=np.zeros(1*(math.factorial(PencereBoyutu)+1))[np.newaxis] #pencere boyutu ayarlanıyor
    elif (MA=="V"):
        Entropiler=np.zeros(257)[np.newaxis]
    else :Entropiler=np.zeros(361)[np.newaxis]
    
    for say in range(2):
        if (say==0):
            folder = r"C:\Users\aidata\Desktop\SpamDataset\enron6\spam"            
            spham=0            
        else : 
            folder = r"C:\Users\aidata\Desktop\SpamDataset\enron6\ham"
            spham=1
        print(folder)
            
        for dosya in os.listdir(folder) :
            uza=os.path.join(folder,dosya)             
            body=readmail(uza)
            if (body!=""):
                Entropiler=np.concatenate([Entropiler,motifinsert(body,spham)[np.newaxis, :]])
    Entropiler = np.delete(Entropiler, (0), axis=0)
    return Entropiler

def ozniteliksecimi(ma,data):
    if (ma=="A"):
        print("Açı Sonuçları")
    elif (ma=="M"):
        print(str(PencereBoyutu)+" : Motif Örüntü Sonuçları")
    else: 
        print(str(alfa)+" alfa, "+str(beta)+" beta  Vektör Sonuçları")
    
    
    df=pd.DataFrame(Only_Motif(ma,data))

    #sonuclar=classification(df)
    #display(sonuclar.sort_values('Accuracy_Score', ascending=False))
    
    X = df.iloc[:,0:len(df.columns)-1]
    Y = df.iloc[:, -1]
    #verilerin egitim ve test icin bolunmesi
    from sklearn.model_selection  import train_test_split
    x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=42)    
    
    x_train=np.array(x_train)
    x_test=np.array(x_test)
    
    train_vecs = np.reshape(x_train, (x_train.shape[0], 1,x_train.shape[1]))
    test_vecs = np.reshape(x_test, (x_test.shape[0], 1,x_test.shape[1]))  
    
    model = Sequential()
    model.add(LSTM(320, return_sequences=True,
                   input_shape=(1,train_vecs.shape[2])))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(320, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(320))  # return a single vector of dimension 32
    model.add(Dense(units=1, activation='sigmoid'))
    
    my_optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=my_optimizer, metrics=['accuracy'])
    model.summary()
    
    mc = keras.callbacks.ModelCheckpoint("best_val_loss", monitor='val_loss', verbose=0, save_best_only=True, mode='min',
                                         save_weights_only=True)
    
    es = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)
    
    model.fit(train_vecs[:-300], y_train[:-300],
              epochs=60,batch_size=2, verbose=2,
             validation_data=(train_vecs[-300:], y_train[-300:]),
             callbacks=[es,mc])
    
    model.load_weights("best_val_loss")
    
    y_pred=model.predict(test_vecs,
                         batch_size=1,
                         verbose=1, steps=None)
    y_pred=y_pred>0.5
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_pred=y_pred,y_true=y_test)
    print('True Positives: ',cm[1,1])
    print('False Positives: ',cm[0,1])
    print('True Negatives: ',cm[0,0])
    print('False Negatives: ',cm[1,0])
    
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(y_test,y_pred)
    print('Accuracy: %f' % accuracy)


def Only_Motif(ma,data):    
    if (ma=="M"):
        Entropiler=np.zeros(1*(math.factorial(PencereBoyutu)+1))[np.newaxis] #pencere boyutu ayarlanıyor
    elif (ma=="V"):
        Entropiler=np.zeros(257)[np.newaxis]
    else :Entropiler=np.zeros(361)[np.newaxis]
           
    x_train = data.iloc[:, :-1].values
    y_train = data.iloc[: ,-1].values     
    df=np.array([])
    for num, satir in enumerate(x_train):
        Entropiler=np.concatenate([Entropiler,motifinsert(ma,satir[0],y_train[num])[np.newaxis, :]])
    Entropiler = np.delete(Entropiler, (0), axis=0)
    return Entropiler

dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Linear Discriminant Analysis(LDA)": LinearDiscriminantAnalysis(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Naive Bayes": GaussianNB(),
    #"K Means": KMeans(n_clusters=3, random_state=0),
    "Neural Net": MLPClassifier(alpha = 1)
    
    #"AdaBoost": AdaBoostClassifier(),
    #"QDA": QuadraticDiscriminantAnalysis(),
    #"Gaussian Process": GaussianProcessClassifier()
    }
def crossvalpredict(): # çapraz doğrulama n-fold
    import nltk
    dosya= open("7all.csv",encoding='utf-8')
    cats=[]
    articles=[]
    vocab=[]
    for line in dosya:
        lines=line.split(",")
        cat=lines[0]
        cats.append(cat)
        article=lines[1].split()
        articles.append(article)
        vocab= vocab+ article
    fd=nltk.FreqDist(vocab)
    from sklearn.feature_extraction.text import CountVectorizer
    K=2000
    mc=fd.most_common(K) 
    freqK=[e[0] for e in mc]
    art2=[" ".join([k for k in a if k in freqK ]) for a in articles]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(art2)
    Y=cats

    for classifier_name, classifier in list(dict_classifiers.items())[:9]:  
        predicted = cross_val_predict(classifier, X, Y, cv=4)
        acc=metrics.accuracy_score(Y, predicted)   
        print(classifier_name+" : acc=",acc)
def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers, verbose = True):
    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        classifier.fit(X_train, Y_train)
        
        pcs=metrics.precision_score(Y_test,classifier.predict(X_test))  #Doğruluk
        fcs=metrics.f1_score(Y_test,classifier.predict(X_test))  #Doğruluk
        acs=metrics.accuracy_score(Y_test,classifier.predict(X_test))  #Doğruluk
        
        dict_models[classifier_name] = {'model': classifier, 'precision_score': round(pcs,4), 'f1_score': round(fcs,4), 'Accuracy_Score':round(acs,4)}
    return dict_models
def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['precision_score'] for key in cls]
    training_s = [dict_models[key]['f1_score'] for key in cls]
    acs_t = [dict_models[key]['Accuracy_Score'] for key in cls]
    
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'precision_score', 'f1_score', 'Accuracy_Score'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'precision_score'] = training_s[ii]
        df_.loc[ii, 'f1_score'] = test_s[ii]
        df_.loc[ii, 'Accuracy_Score'] = acs_t[ii]
    
    return df_
def classification(df):
    X = df.iloc[:,0:len(df.columns)-1]
    Y = df.iloc[:, -1]
    #verilerin egitim ve test icin bolunmesi
    from sklearn.model_selection  import train_test_split
    x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=42)    
    
    dict_models = batch_classify(x_train, y_train, x_test, y_test, 9)
    sonuclar=pd.DataFrame(display_dict_models(dict_models))
    return sonuclar


PencereBoyutu=4

komsuluk=4
alfa=1
beta=1
komsuluk=komsuluk*beta

data = pd.read_csv(r'D:\Dropbox\Doktora\veriseti\yemeklerin_sepeti\yemeklerin_sepeti.csv')
data=data.dropna()
data = data.drop_duplicates(subset=['yorum'])
data=data.reset_index()
data=data.drop('index',axis=1)
for i in range(len(data.hiz)):
    data.hiz[i]=re.sub("\D", "", data.hiz[i])
for i in range(len(data.servis)):
    data.servis[i]=re.sub("\D", "", data.servis[i])
for i in range(len(data.lezzet)):
    data.lezzet[i]=re.sub("\D", "", data.lezzet[i])

data['hiz'] = data['hiz'].astype(int)
data['servis'] = data['servis'].astype(int)
data['lezzet'] = data['lezzet'].astype(int)
data.loc[data['hiz']<4, 'hiz'] = 0
data.loc[data['hiz'] >7, 'hiz'] = 1
data.loc[data['lezzet']<4, 'lezzet'] = 0
data.loc[data['lezzet'] >7, 'lezzet'] = 1
data.loc[data['servis']<4, 'servis'] = 0
data.loc[data['servis'] >7, 'servis'] = 1

data=data[(data['hiz']<2) & (data['servis']<2) & (data['lezzet']<2)]
data_positive=data[(data['hiz']==1) & (data['servis']==1) & (data['lezzet']==1)]
data_negative=data[(data['hiz']==0) & (data['servis']==0) & (data['lezzet']==0)]
data_positive=data_positive[:600]
data=pd.concat([data_positive,data_negative],axis=0)
data=data.drop_duplicates()

data['isPositive']=np.where(((data['hiz']==1) & (data['servis']==1) & (data['lezzet']==1)),1,0)
data=data.drop(["hiz","servis","lezzet"],axis=1)

ozniteliksecimi("M",data)
ozniteliksecimi("A",data)
ozniteliksecimi("V",data)
    
# y_test = y_test.reset_index(drop = True)
# y_test=np.reshape(y_test,(len(y_test,)))
# y_pred=np.reshape(y_pred,(len(y_pred,)))

# x_test = pd.Series(x_test)
# y_test=pd.Series(y_test)
# y_pred=pd.Series(y_pred)
# inspection = pd.concat([x_test, y_test, y_pred], axis=1)

# inspection.columns=['Yorum', 'Test','Predictions']

# wrong_predictions = inspection[inspection["Predictions"] != inspection["Test"]]
# print(wrong_predictions)
#file_name=r'C:\Users\aidata\Desktop\SpamDataset\enron6\Enron7Aci.csv'
#df.to_csv(file_name, encoding='utf-8',index=False, header=False)


    
    
    
    
    
    
    