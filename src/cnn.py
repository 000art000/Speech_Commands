import numpy as np 
import os
import os 
import pickle
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from os import listdir
from os.path import isfile

import torch
from torch.utils.data import DataLoader,TensorDataset
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from segmentation import vote

class M1(nn.Module):
    
    def __init__(self,param):
        super(M1, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=30, kernel_size=16)
        self.bn1 = nn.BatchNorm1d(30)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(in_channels=30, out_channels=30 ,kernel_size=3)
        self.bn2 = nn.BatchNorm1d(30)
        self.pool2 = nn.MaxPool1d(4)
        
        self.lin1 = nn.Linear(param, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 128)
        self.lin4 = nn.Linear(128, 256)
        self.lin5 = nn.Linear(256, 35)

        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
            x = self.conv1(x) 
            x = F.relu(self.bn1(x))
            x = self.pool1(x)
        
            x = self.conv2(x)
            x = F.relu(self.bn2(x))
            x = self.pool2(x)

            x = x.view(x.size(0), -1)
            x = self.lin1(x)
            x = F.tanh(x)
            x = self.lin2(x)
            x = F.tanh(x)
            x = self.lin3(x)
            x = F.tanh(x)
            x= self.lin4(x)
            x = F.tanh(x)
            x = self.lin5(x)
            x = self.log_softmax(x)
        
            return x
    
    def predict(self,x,device):
        x = torch.tensor(x,device=device)
        x =  torch.reshape(x, (x.shape[0],1,x.shape[1]))
        x = torch.tensor(x, device=device, dtype=torch.float32)

        y_hat=None

        for debut in range(0, len(x), 1024):
            fin = debut + 1024
            bloc = x[debut:fin]

            y=self.forward(bloc)
            y=torch.argmax(y, dim=1) 

            if y_hat==None:
                y_hat=y
            else :
                y_hat= torch.cat((y_hat, y), dim=0)
                        
        return y_hat
             
    
def train(model, train_loader, optimizer,loss_f, epoch, device):
        model.train()
    
        for _ in tqdm(range(epoch)): 
        
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
        
                data =  torch.reshape(data, (data.shape[0],1,data.shape[1]))

                data = torch.tensor(data, device=device, dtype=torch.float32)

                optimizer.zero_grad()
        
                output = model(data)

                loss = loss_f(output, target)
        
                loss.backward()
        
                optimizer.step()

def predict(model,path,mode,train,t,device):
    typ='train' if train else 'test'
    print('predict',t)
    x_train=np.load(f'{path}/{typ}_{t}_{mode}.npy')
    label=np.load(f'{path}/{typ}_label.npy')
    codage_train=np.load(f'{path}/codage_{typ}_{t}.npy')

    y_hat=model.predict(x_train,device).numpy()
    label_hat=vote(y_hat,codage_train)

    r1=balanced_accuracy_score(label,label_hat)

    r2=None
    if train :
        r2=confusion_matrix(label,label_hat)

    return r1,r2

def stock_result(model,path,mode,path_stock,device,chauvechement=False):
    t='sc' if chauvechement==False else 'ac'
    print('stock_result ',t)
    s_test,c_test=predict(model,path,mode,False,t,device)
    s_train,c_train=predict(model,path,mode,True,t,device)

    print(s_test,s_train)

    dir=path_stock.split('/')
    
    if not os.path.exists(dir[0]):
        os.mkdir(dir[0])
    if not os.path.exists(dir[0]+f'/{dir[1]}'):
        os.mkdir(dir[0]+f'/{dir[1]}')
    if not os.path.exists(dir[0]+f'/{dir[1]}/{dir[2]}'):
        os.mkdir(dir[0]+f'/{dir[1]}/{dir[2]}')
    if not os.path.exists(dir[0]+f'/{dir[1]}/{dir[2]}/{dir[3]}'):
        os.mkdir(dir[0]+f'/{dir[1]}/{dir[2]}/{dir[3]}')
    if not os.path.exists(path_stock):
        os.mkdir(path_stock)


    results = {
	    'best_score': s_train,
	    'test_score': s_test,
	    'confusion_matrix': c_train,
	}

    
    # Écriture des résultats dans un fichier pickle
    with open(path_stock+'/results_'+ t + '_' + mode + '.pickle', 'wb') as f:
        pickle.dump(results, f)
     


def get_data_loader(batch_size,num_workers,pin_memory,path,mode,shuffle=True,train=True,chauvechement=False):

    typ='train' if train else 'test'
    t='sc' if chauvechement==False else 'ac'
    print('get_data_loader ',t)
    x=np.load(f'{path}/{typ}_{t}_{mode}.npy')

    if train :
        y=np.load(f'{path}/label_{t}.npy')

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    dataset = TensorDataset(x, y)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,pin_memory=pin_memory)

def lanch(batch_size,out_put_conv,epochs,num_workers,pin_memory,mode,device,typ_data,chauvechement=False):
    train_loder=get_data_loader(batch_size,num_workers,pin_memory,f'dataset/segmentation/{typ_data}',mode,chauvechement=chauvechement)
    if chauvechement :
        Y_train=np.load(f'dataset/segmentation/{typ_data}/label_ac.npy')
        print(chauvechement)
    else :
        Y_train=np.load(f'dataset/segmentation/{typ_data}/label_sc.npy')

    _,priors=np.unique(Y_train,return_counts=True)
    priors= 1/torch.tensor(priors ,dtype=torch.float32)

    model = M1(out_put_conv).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Specify the loss criteria
    loss_criteria = nn.CrossEntropyLoss(weight=priors)

    train(model,train_loder,optimizer,loss_criteria,epochs,device)

    stock_result(model,f'dataset/segmentation/{typ_data}',mode,f'resultat/cnn/{typ_data}/2_conv/5_lineare',device,chauvechement=chauvechement)


def affichage(seg):
    m=[]
    r=[]
    path = f"resultat/cnn/{seg}/"
    score_app=[]
    score_test=[]
    list_chauv=[]
                
    for conv in ['2_conv']:
        for representation in ['mfcc']:
            for modele in ['1_lineare','2_lineare','3_lineare','5_lineare']:
                for chau in ['sc','ac']:
                    new_path=path
                    new_path+=conv
                    new_path+=f'/{modele}'
                    for f in listdir(new_path):

                        if isfile(f'{new_path}/{f}') and 'pickle' in f and representation in f and chau in f:
                            r.append(f'{representation}')
                            m.append(conv+' '+modele)
                            with open (f'{new_path}/{f}','rb') as fd :
                                obj=pickle.load(fd)
                                score_test.append(obj['test_score'])
                                score_app.append(obj['best_score'])
                                list_chauv.append('oui' if chau=='ac' else 'non')

    data={'representation':r,'modele':m,'chauvechement':list_chauv,'score_app':score_app,'score_test':score_test}
    return pd.DataFrame(data)

def show_matrix(modele,chauvechement,representation,typ_data,conv):
    def aux(r,c,l):
        for ele in l:
            if r  in ele  and c in ele :
                return ele
            
    path = f"resultat/cnn/{typ_data}/{conv}/{modele}"
    onlyfiles = [f for f in listdir(path) if isfile(f'{path}/{f}') ]
    f=aux(representation,chauvechement,onlyfiles)

    plt.figure(figsize=(11, 8))

    with open (f'{path}/{f}','rb') as fd :
        obj=pickle.load(fd)
        mat_conf=obj['confusion_matrix']
        sns.heatmap(mat_conf, annot=True, cmap="Blues", fmt="d", cbar=False)
        # Configuration des labels des axes
        plt.xlabel("Classe prédite")
        plt.ylabel("Classe réelle")

        # Affichage du graphique
        plt.show()