import numpy as np 
import librosa 
import os
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import math 
import json
import re
from os import listdir
from os.path import isfile
from functools import reduce
import pickle
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


# defintion des chemin de fichier d"app et de test
file_app='validation_list.txt'
file_test='testing_list.txt'



# fonction qui renvoie la taille de la max et min des signals
def taille_signal(path_file,data_path, approche=True):
    """
        fonction permettant de déterminer l'audio le plus court/long.

        Args:
            path_file (str): chemin du fichier d'app
            data_path (str): chemin du data brute.
            approche (booleen) : qu'elle approche on utilise (1/2)

        Returns:
            int : la taille de l'audio le plus court/long. 

    """
    res = -np.inf if approche else np.inf
    with open(path_file,'r') as fd :
        for line in fd :
            signal,sr=librosa.load(data_path+'/'+line.rstrip())
            res = max(res, signal.shape[0]) if approche else min(res, signal.shape[0])
            
    return res


# fonction qui genere le dataset
def stock_data(data_path,path_data_save,mode, approche=True,apprentissage=True):
    """"
        args :
            data_path : chemin du rep des donnée brute
            path_data_save : chemin du rep pour stocké les fichier de dataset
            mode : mfcc ou brute ....
            approche : true -> 1er , false -> 2eme
            apprentissage : donner test ou app generer
    """

    X=[]
    Y=[]

    path_file = file_app if apprentissage else file_test
    path_file= data_path+'/'+ path_file

    max_taille=taille_signal(data_path+'/'+ file_app,data_path,approche)

    
    with open(path_file,'r') as fd :
        for line in (fd) :
            label=line.split('/')[0]
            signal,sr=librosa.load(data_path+'/'+line.rstrip())
            
            if approche:
                if signal.shape[0]<max_taille:
                    signal=np.hstack( ( np.reshape(signal,(1,-1)) , np.zeros((1,max_taille-signal.shape[0])) ))
                    signal=np.reshape(signal,(-1,))
            else:
                # pour le test car ya des signal sont plus petit que ceux d'entrainement
                if signal.shape[0]<max_taille :
                     signal=np.hstack( ( np.reshape(signal,(1,-1)) , np.zeros((1,max_taille-signal.shape[0])) ))
                signal=signal[:max_taille]
                signal=np.reshape(signal,(-1,))

            
            if mode=='brute':
                X.append(signal)    
            elif mode=='harm':
                y_harm, _ = librosa.effects.hpss(signal)
                X.append(y_harm)
            elif mode=='perc' :
                _, y_perc = librosa.effects.hpss(signal)
                X.append(y_perc)
            elif mode=='spec':
                spec=librosa.stft(signal)
                spec=np.reshape(spec,(-1,))
                X.append(spec)
            elif mode=='mfcc':
                mfcc=librosa.feature.mfcc(y=signal, sr=sr)
                mfcc=np.reshape(mfcc,(-1,))
                X.append(mfcc)
            elif mode=='melspec':
                melspec=librosa.feature.melspectrogram(y=signal, sr=sr)
                melspec=np.reshape(melspec,(-1,))
                X.append(melspec)

            Y.append(label)

    X,Y_to_encode=np.array(X),np.array(Y)

    le=LabelEncoder()
    Y=le.fit_transform(Y_to_encode)

    if not os.path.exists(path_data_save):
        os.mkdir(path_data_save)

    approche='/max' if approche else '/min'

    if not os.path.exists(path_data_save+approche):
        os.mkdir(path_data_save+approche)

    typ= '/train' if apprentissage else '/test'

    np.save(path_data_save+approche+ typ +'_data_'+mode,X)
    np.save(path_data_save+approche+typ+'_label',Y)



# fonction qui charge le dataset
def load_data(dir_path,mode,approche=True,apprentissage=True):
    approche='/max' if approche else '/min'
    typ= '/train' if apprentissage else '/test'
    X=np.load(dir_path+approche+typ+'_data_'+mode+'.npy')
    Y=np.load(dir_path+approche+typ+'_label.npy')
    return X,Y



## fonction qui fait l'apprentissage
def learning(grid,rep_save,mod,X,Y,X_test,Y_test,modele,approche='max'):
	grid.fit(X,Y)
    
	results = {
	    'best_score': grid.best_score_,
	    'best_estimator': grid.best_estimator_,
	    'test_score': balanced_accuracy_score(Y_test,grid.best_estimator_.predict(X_test)),
	    'confusion_matrix': confusion_matrix(Y, grid.best_estimator_.predict(X)),
	}
        
	if not os.path.exists(rep_save):
		os.mkdir(rep_save)
	if not os.path.exists(rep_save+f'/{approche}'):
		os.mkdir(rep_save+f'/{approche}')
	with open(rep_save+f'/{approche}/'+mod+f'_{modele}'+ '.pickle', 'wb') as f:
		pickle.dump(results, f)
