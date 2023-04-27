import numpy as np 
import librosa 
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import os 
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

# defintion des chemin de fichier d"app et de test
file_app='validation_list.txt'
file_test='testing_list.txt'

sr_global=None

def sans_chauvechement(sr,size_w,cpt,signal):
        """
            args:
                sr: frequence d'echantillonage
                size_w : taille de la fenetre en ms
                cpt : variable qui marque le debut ou la fin d'un signal lors de la segmentation
                signal : notre signal
        """
        l=[]
        #taille du segment (taille de la fenetre en ms en systeme temporale)
        size_fr= sr*size_w//10**3
        j=0

        # rajouter dans le signal des 0 si il est plus petit que la fenetre
        if signal.shape[0]<size_fr:
            signal=np.hstack( ( signal , np.zeros((size_fr-signal.shape[0]),) ))

        # on prend pas le dernier segment lors du decoupage
        while len(signal)>=size_fr :
            #mettre notre fragment sous format:  notre fragment, pos entre les fragment des echantillons 
            fr=signal[:size_fr]
            fr = np.concatenate((fr,[j]),axis=0)
            l.append(fr)
            j+=1
            signal=signal[size_fr:]

        cpt+=j

        return cpt,l
    
def avec_chauvechement(sr,size_w,cpt,signal,chauvechement):
        """
            args:
                sr: frequence d'echantillonage
                size_w : taille de la fenetre en ms
                cpt : variable qui marque le debut ou la fin d'un signal lors de la segmentation
                signal : notre signal
                chauvechement : la taille de chauvechement qu'on veut en ms 
        """
        l=[]
        #taille du segment (taille de la fenetre en ms en systeme temporale)
        size_fr= sr*size_w//10**3
        #ofset a faire pour respecter le chauvechement voulu
        size_c= sr*(size_w-chauvechement)//10**3

        # rajouter dans le signal des 0 si il est plus petit que la fenetre
        if signal.shape[0]<size_fr:
            signal=np.hstack( ( signal , np.zeros((size_fr-signal.shape[0]),) ))
        
        j=0
        # on prend pas le dernier segment lors du decoupage
        while signal.size>=size_fr :
            #mettre notre fragment sous format:  notre fragment, pos entre les fragment des echantillons 
            fr=signal[:size_fr]
            fr = np.concatenate((fr,[j]),axis=0)
            l.append(fr)
            j+=1
            signal=signal[size_c:]
        
        cpt+=j


        return cpt,l


def brute(X):
    return X

def harm(X):
    x_temp=[]
    for fragment in X:
        x_temp.append(librosa.effects.harmonic(fragment))
    return x_temp

def pers(X):
    x_temp=[]
    for fragment in X:
        x_temp.append(librosa.effects.percussive(fragment))
    return x_temp 

def mfcc(X):
    x_temp=[]
    for fragment in X:
        x_temp.append( np.reshape(librosa.feature.mfcc(y=fragment, sr=sr_global, n_fft= fragment.size),(-1,))  )
    return x_temp      

def melspec(X):
    x_temp=[]
    for fragment in X:
        x_temp.append( np.reshape(librosa.feature.melspectrogram(y=fragment,sr=sr_global,n_fft=fragment.size),(-1,))  )
    return x_temp         

def spec (X):
    x_temp=[]
    for fragment in X:
        x_temp.append( np.reshape(librosa.stft(fragment,n_fft=fragment.size),(-1,))  )
    return x_temp 


def stock_data(data_path,path_data_save,mode,size_w,apprentissage=True,chauvechement=None):

    """
    args  :
        data_path : chemin pour trouver les fichier brute
        path_data_save : rep pour sauvgarder le dataset
        mode : mfcc ....
        size_w : taille du fragment qu'on veut en ms
        chauvechement : taille du chauvechement des fragments qu'on veut en ms
    """

    X=[]
    Y=[]
    labels=[]

    X_codage=[]

    global sr_global

    path_file = file_app if apprentissage else file_test
    path_file= data_path+'/'+ path_file

    if mode =='brute':
        transformation=brute
    elif mode =='harm':
        transformation=harm
    elif mode=='pers':
        transformation = pers
    elif mode=='mfcc':
        transformation = mfcc
    elif mode=='spec':
        transformation = spec
    elif mode=='melspec':
        transformation = melspec
    else : 
        print('erreur de mode')
        assert Exception


    with open(path_file,'r') as fd :
            # sert pour le debut et la fin d'un echantillons pour indexer ses segments
            cpt=0

            for line in  fd :
                label=line.split('/')[0]
                signal,sr=librosa.load(data_path+'/'+line.rstrip())  

                sr_global=sr

                # position du premier fragment de ce singal
                X_codage.append([cpt])

                if chauvechement==None:
                    cpt,l=sans_chauvechement(sr,size_w,cpt,signal)
                else :
                    cpt,l=avec_chauvechement(sr,size_w,cpt,signal,chauvechement)

                #position de fin du dernier fragment de ce singal 
                X_codage[-1].append(cpt)

                # les labels des des fragments pour l'apprentissage
                if apprentissage :
                    Y.extend( [ label for _ in range(cpt-X_codage[-1][0])] )

                # les lables des signaux
                labels.append(label)
                
                # les signaux fragmenter
                X.extend(transformation(l))


            X=np.array(X)    
            labels=np.array(labels)
            Y=np.array(Y)

            le=LabelEncoder()
            labels=le.fit_transform(labels)

            if not os.path.exists(path_data_save):
                os.mkdir(path_data_save)
            
            path_data_save=f'{path_data_save}/segmentation'

            if not os.path.exists(path_data_save):
                os.mkdir(path_data_save)

            path_data_save=f'{path_data_save}/{size_w}'

            if not os.path.exists(path_data_save):
                os.mkdir(path_data_save)

            typ='train' if apprentissage else 'test'
            t='sc' if chauvechement==None else 'ac'
            
            np.save(f'{path_data_save}/{typ}_{t}_{mode}',X)
            np.save(f'{path_data_save}/{typ}_label',labels)
            np.save(f'{path_data_save}/codage_{typ}_{t}',X_codage)

            if apprentissage :
                Y=le.transform(Y)
                np.save(f'{path_data_save}/label_{t}',Y)

### function qui se charge de charger les deffrent fichier qu'on a besoin      
def load_data(path,mode,apprentissage=True,chauvechement=None):
    typ='train' if apprentissage else 'test'
    t='sc' if chauvechement==None else 'ac'

    x=np.load(f'{path}/{typ}_{t}_{mode}.npy')
    label=np.load(f'{path}/{typ}_label.npy')
    y=None        
    codage=np.load(f'{path}/codage_{typ}_{t}.npy')

    if apprentissage :
        y=np.load(f'{path}/label_{t}.npy')

    return x,y,codage,label


def vote(Y_hat,codage):
    # Transformation des données Y_hat
        res=np.zeros((codage.shape[0],))

        for i,ind in enumerate(codage):
            debut,fin=ind[0],ind[1]
            y,occ=np.unique(Y_hat[debut:fin], return_counts=True)
            res[i]=y[np.argmax(occ)]

        return res

#function qui fait l'apprentissage
def learning(grid, X_train, Y_train,y_label,X_test,Y_test,codage_train,codage_test,modele,technique,mode,rep_save):
    print('fit')
    grid.fit(X_train, Y_train)
    print('prediction')
	# Utilisation de la sortie de la grille de recherche
    pred_test=vote(grid.predict(X_test),codage_test)
    pre_train=vote(grid.predict(X_train),codage_train)
    print('prediction done')

    results = {
	    'best_score': balanced_accuracy_score(y_label,pre_train),
	    'best_params': grid.best_params_,
	    'best_estimator': grid.best_estimator_,
	    'test_score': balanced_accuracy_score(Y_test,pred_test),
	    'confusion_matrix': confusion_matrix(y_label, pre_train),
	}
    
    dir=rep_save.split('/')
    
    if not os.path.exists(dir[0]):
        os.mkdir(dir[0])
    if not os.path.exists(dir[0]+f'/{dir[1]}'):
        os.mkdir(dir[0]+f'/{dir[1]}')
    if not os.path.exists(rep_save):
        os.mkdir(rep_save)

	# Écriture des résultats dans un fichier pickle
    with open(rep_save+'/results_'+modele+'_'+ technique + '_' + mode + '.pickle', 'wb') as f:
        pickle.dump(results, f)
                          
                        
def lanch(path_data,rep_save,mode):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    ###############################################################################################################################

    #sans chauvechement
    X_train, Y_train,codage_train,label_train=load_data(path_data,mode)
    X_test, Y_test,codage_test,label_test=load_data(path_data,mode,apprentissage=False)

    _,priors=np.unique(Y_train,return_counts=True)
    priors=priors/priors.sum()
    
    #nb
    param_nb = {'var_smoothing': [1e-9,1e-8,1e-7,1e-6]}
    grid_search = GridSearchCV(GaussianNB(priors=priors), param_nb, cv=skf,scoring='balanced_accuracy')
    learning(grid_search, X_train, Y_train,label_train,X_test,label_test,codage_train,codage_test,"nb","sc",mode,rep_save)

    #knn
    param_grid = {'n_neighbors' : np.arange(5,15,1)}
    grid_knn = GridSearchCV(KNeighborsClassifier(n_jobs=-1), param_grid, cv=skf,scoring='balanced_accuracy')
    learning(grid_knn, X_train, Y_train,label_train,X_test,label_test,codage_train,codage_test,"knn","sc",mode,rep_save)

    
    ##################################################################################################################################
    

    #sans chauvechement
    X_train, Y_train,codage_train,label_train=load_data(path_data,mode,chauvechement=True)
    X_test, Y_test,codage_test,label_test=load_data(path_data,mode,apprentissage=False,chauvechement=True)

    _,priors=np.unique(Y_train,return_counts=True)
    priors=priors/priors.sum()

    #knn
    param_grid = {'n_neighbors' : np.arange(5,15,1)}
    grid_knn = GridSearchCV(KNeighborsClassifier(n_jobs=-1), param_grid, cv=skf,scoring='balanced_accuracy')
    learning(grid_knn, X_train, Y_train,label_train,X_test,label_test,codage_train,codage_test,"knn","ac",mode,rep_save)

    #nb
    param_nb = {'var_smoothing': [1e-9,1e-8,1e-7,1e-6]}
    grid_search = GridSearchCV(GaussianNB(priors=priors), param_nb, cv=skf,scoring='balanced_accuracy')
    learning(grid_search, X_train, Y_train,label_train,X_test,label_test,codage_train,codage_test,"nb","ac",mode,rep_save)


