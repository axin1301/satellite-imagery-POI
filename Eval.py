import torch
from torch import nn
import torch.nn.functional as F
import math
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import shutil
import random
import numpy as np
from sklearn import manifold
from sklearn.decomposition import PCA
from collections import Counter
import pandas as pd
import os
import setproctitle
import pdb
from torch.autograd import Variable
import warnings
from sklearn import metrics
os.environ["CUDA_VISIBLE_DEVICES"]="7"
setproctitle.setproctitle('1')
    

def weights_init_1(m):
    seed=20
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.nn.init.xavier_uniform_(m.weight,gain=1)
    
def weights_init_2(m):
    seed=20
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.nn.init.xavier_uniform_(m.weight,gain=1)
    torch.nn.init.constant_(m.bias,0)


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()

        self.l1=torch.nn.Linear(in_size, hidden_size, bias=True)
        self.ac=nn.Tanh()
        self.l2=torch.nn.Linear(int(hidden_size), 1, bias=False)

        weights_init_2(self.l1)
        weights_init_1(self.l2)
        

    def forward(self, z):
        w=self.l1(z)
        w=self.ac(w)
        w=self.l2(w)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)




class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear1 = torch.nn.Linear(inputSize, inputSize, bias=True)
        self.act1 = nn.ReLU()
        self.linear2 = torch.nn.Linear(inputSize, outputSize, bias=True)

        self.attention = Attention(in_size = inputSize)
        weights_init_2(self.linear1)
        weights_init_2(self.linear2)
        
    def forward(self, x1, x2):#
        Features = torch.stack([x1, x2], dim=1)
        Features = self.attention(Features)

        out = self.linear1(Features)
        out=self.act1(out)
        out = self.linear2(out)
        
        return out

    
    
if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    #load data
    image_name_POI=list(pd.read_csv('img_name_all.csv',header=0,sep=',')['img_name'])#satellite image list: ID, an one-column csv file
    feature_all1=np.loadtxt('data_feature_POI.txt') #embeddings of satellite images from POI-nearest model, a numpy array: image number * feature length
    feature_all2=np.loadtxt('data_feature_spatial.txt') #geo-nearest embeddings of satellite images
    
    image_data=list(pd.read_csv('dianping_Sat.csv',header=0,sep=',')['SECOND_FLD']) #comment data: a two-column csv file, first column: image name; second column: socioeconomic indicator
    wm_data=list(pd.read_csv('dianping_Sat.csv',header=0,sep=',')['review-count'])
    
    wm_data_log = [np.log(item+1) for item in wm_data] # log transform
    fea_1=np.zeros((len(image_data),feature_all1.shape[1]))
    fea_2=np.zeros((len(image_data),feature_all2.shape[1]))
    
    for i in range(len(image_data)): # in case some satellite images are missing groundtruth values, they are deleted from the evaluation
        tmp_im=image_data[i]
        POI_idx=image_name_POI.index(tmp_im)
        fea_1[i,:]=feature_all1[POI_idx,:]
        fea_2[i,:]=feature_all2[POI_idx,:]
    
    input_data1=fea_1
    input_data2=fea_2
    output_data=np.array(wm_data_log)


    x=np.arange(0,input_data2.shape[0])
    idx_train,idx_test,y_train,y_test= \
    train_test_split(x,output_data,test_size=0.2,random_state=100) 

    idx_train,idx_val,y_train,y_val= \
    train_test_split(idx_train,y_train,test_size=0.25,random_state=100)

    
    x_train1=input_data1[idx_train,:]
    x_train2=input_data2[idx_train,:]
    x_train1 = torch.as_tensor(x_train1, dtype=torch.float32).cuda()
    x_train2 = torch.as_tensor(x_train2, dtype=torch.float32).cuda()
    y_train = torch.as_tensor(y_train.reshape((-1, 1)), dtype=torch.float32).cuda()

    x_val1=input_data1[idx_val,:]
    x_val2=input_data2[idx_val,:]
    x_val1 = torch.as_tensor(x_val1, dtype=torch.float32).cuda()
    x_val2 = torch.as_tensor(x_val2, dtype=torch.float32).cuda()
    y_val = torch.as_tensor(y_val.reshape((-1, 1)), dtype=torch.float32)


    x_test1=input_data1[idx_test,:]
    x_test2=input_data2[idx_test,:]
    x_test1 = torch.as_tensor(x_test1, dtype=torch.float32).cuda()
    x_test2 = torch.as_tensor(x_test2, dtype=torch.float32).cuda()
    y_test = torch.as_tensor(y_test.reshape((-1, 1)), dtype=torch.float32)


    val_tmp=0
    for learningRate in [0.01]
        inputDim = fea_1.shape[1]  
        outputDim = 1
        epochs = 400
    
        model = linearRegression(inputDim, outputDim)
        model.cuda()
    
        criterion = torch.nn.MSELoss() 
        optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate,weight_decay=0.01)
        
    
        for epoch in range(epochs):    
            model.train()
            outputs = model(x_train1, x_train2)
            loss = criterion(outputs, y_train)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad(): 
                predicted = model(x_train1,x_train2).cpu()
                r2_train=r2_score(list(y_train), list(predicted))
                
                
            with torch.no_grad(): 
                model.eval()
                predicted = model(x_val1, x_val2).cpu()
                r2_val=r2_score(list(y_val), list(predicted))
            
            with torch.no_grad(): 
                model.eval()
                predicted = model(x_test1,x_test2).cpu().data.numpy()
                r2=r2_score(list(y_test), list(predicted))
                RMSE=np.sqrt(mean_squared_error(list(y_test), list(predicted)))
                MAE=metrics.mean_absolute_error(list(y_test), list(predicted))
                
            if val_tmp<r2_val:
                #torch.save(model.state_dict(),'dif_file3/'+str(1)+'.ckpt')
                val_tmp=r2_val
                print('Epoch:', epoch, 'Train loss:', loss)
                print('Train_R2: ',r2_train)
                print('Val_R2: ',r2_val)
                print('Test_R2: ',r2)
                print('RMSE',RMSE)
                print('MAE ',MAE)
      
