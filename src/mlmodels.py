# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 23:02:35 2025

"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_PRINTED_SETUP = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.random.seed(5723588)
torch.manual_seed(5723588)
YLAB=['W','Ed','Home','O','So','Sh','Work','Others','Home','Shop'] #first 6: round 1 remainings: round 2 from pd.factorize()
USECUML=False

if __name__=='__main__':
    if '__file__' in globals():
        from pathlib import Path
        WPATH = Path(__file__).resolve().parent.parent #.parent=src .parent.parent=MLTraining
    elif os.name=='posix': #linux server; Windows='nt'
        WPATH='/export/scratch/users/baek0040/git/TCML/MLTraining'
    else:
        if os.environ['USERPROFILE']==r'C:\Users\baek0040':
            WPATH='C:/Users/baek0040/Documents/GitHub/Trip-Chaining-and-Machine-Learning-Inference/MLTraining'
        else:
            WPATH=os.path.abspath('C:/git/Trip-Chaining-and-Machine-Learning-Inference/MLTraining')
    os.chdir(WPATH)

if os.name=='posix' and os.getenv('USER')=='sh222': #if run on WSL2-CUML compatible
    import cudf
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.svm import SVC as cuSVC
    USECUML=True

def modelResults(yraw,yhat,modelName='',ylab=YLAB):
    print(f"Classification Report for {modelName}:")
    if yraw.max()==5: #detect if round 1
        ylab=ylab[:6]
    else: #round 2
        ylab=ylab[6:]
    print(classification_report(yraw, yhat, target_names=ylab,digits=3, zero_division=np.nan))
    report=classification_report(yraw, yhat, target_names=ylab,digits=4, output_dict=True, zero_division=0)
    accuracy=report['accuracy']
    stats={k: v for k, v in report.items() if k not in ["accuracy", "macro avg", "weighted avg"]}
    stats=pd.DataFrame(stats).transpose()
    flattened=stats.stack()
    #flattened=confusion.loc[np.intersect1d(['Home','W','Work'],ylab)].drop(columns='support').stack()
    flattened.index = flattened.index.map(lambda x: f"{x[0]}_{x[1]}")
    flattened['accuracy']=accuracy
    flattened['zeroPreds']=sum(stats.recall==0) #precision can be div0 if there's no label prediction
    if 'Shop' in ylab:
        confusion=confusion_matrix(yraw,yhat)
        alreadyTrue=np.trace(confusion)
        crossTrue=confusion[np.array(ylab)=='Others',np.array(ylab)=='Shop'][0]+confusion[np.array(ylab)=='Shop',np.array(ylab)=='Others'][0]
        adjAcc=(alreadyTrue+crossTrue)/confusion.sum()
        flattened['adj_accuracy']=adjAcc
    return flattened
#yraw, yhat, ylab= y2.copy(), yhat2.copy(), YLAB

def threadAllocation(deviceUsing='CpU'): #for Catboost and XGBoost; for NN, always use GPU if available: 10-20x faster than GPU
    maxCPUConsole=1+(os.name=='posix')
    if deviceUsing.lower()=='cpu': #XG needs either cuda/cpu and CB needs GPU/CPU in their respective arguments
        if os.name=='nt': #windows
            threadCount=os.cpu_count()-4
            print(f'This console does not use a GPU (CPU usage limited to {threadCount} threads); do not parallel run other CPU consoles')
        else: #linux, WSL, and possibly Mac?
            threadCount=int((os.cpu_count()-2)/2)
            print(f'This console does not use a GPU (CPU usage limited to {threadCount} threads); you may run 1 GPU and 2 CPU consoles simultaneously')
    else: #gpu
        threadCount=os.cpu_count()-2
        print(f'This console uses a GPU and can access 2 CPU threads; you may run at most {maxCPUConsole} CPU console(s) parallely in addition to this console')
    return threadCount


def doRF(x,y,testBool=None,nTrees=200, paramsIn:dict=None):
    clf_func = RandomForestClassifier # Default to sklearn's RandomForest
    if USECUML:
        paramsIn['n_streams']=1
        x=cudf.DataFrame.from_records(x) # Convert to cudf DataFrame for cuML
        y=cudf.Series(y) # Convert to cudf Series for cuML
        clf_func=cuRF # Switch to cuML's RandomForest
        print ('Using cuML')
    if testBool is not None:
        x_ts=x[testBool]
        y_ts=y[testBool]
        x=x[~testBool]
        y=y[~testBool]
    if paramsIn is None:
        rf_classifier=clf_func(n_estimators=nTrees,random_state=5723588)
    else:
        rf_classifier=clf_func(**paramsIn,random_state=5723588)
    rf_classifier.fit(x, y)
    yhat=rf_classifier.predict(x)
    if testBool is None:
        if USECUML:
            yhat=yhat.to_numpy()
        return rf_classifier, yhat
    else:
        yhat_ts = rf_classifier.predict(x_ts)
        if USECUML:
            y_ts=y_ts.to_numpy()
            yhat_ts=yhat_ts.to_numpy()
        return modelResults(y_ts,yhat_ts,'RF')

def doXG(x,y,testBool=None,nTrees=200,tryGPU=True,paramsIn:dict=None):
    XGDevice = 'cuda' if torch.cuda.is_available() and tryGPU else 'cpu'
    threadCount=threadAllocation(XGDevice)
    if testBool is not None:
        x_ts=x[testBool]
        y_ts=y[testBool]
        x=x[~testBool]
        y=y[~testBool]
    if paramsIn is None:
        xg_classifier=xgb.XGBClassifier(n_estimators=nTrees,tree_method='hist',device=XGDevice,
                                        n_jobs=threadCount,random_state=5723588)
    else:
        xg_classifier=xgb.XGBClassifier(**paramsIn,tree_method='hist',device=XGDevice,
                                        n_jobs=threadCount,random_state=5723588)
    xg_classifier.fit(x, y, verbose=0)
    yhat=xg_classifier.predict(x)
    if testBool is None:
        return xg_classifier, yhat
    else:
        yhat_ts = xg_classifier.predict(x_ts)
        return modelResults(y_ts,yhat_ts,'XG')

def doSV(x, y,testBool=None, kernel='linear', paramsIn:dict=None):
    clf_func=SVC
    doCUML=((USECUML) and (paramsIn is not None) and (paramsIn['kernel']!='linear'))
    if doCUML:
        x=cudf.DataFrame.from_records(x)
        y=cudf.Series(y)
        clf_func=cuSVC
        print ('Using cuML')
    if testBool is not None:
        x_ts=x[testBool]
        y_ts=y[testBool]
        x=x[~testBool]
        y=y[~testBool]
    if paramsIn is None:
        clf = clf_func(kernel=kernel,random_state=5723588)
    else:
        clf = clf_func(**paramsIn,random_state=5723588)
    #clf=clf_func(kernel='poly',C=1,class_weight='balanced',degree=3,coef0=0.1,gamma=0.9,random_state=5723588)
    clf.fit(x, y)
    yhat = clf.predict(x)
    if testBool is None:
        if doCUML:
            yhat=yhat.to_numpy()
        return clf, yhat
    else:
        yhat_ts = clf.predict(x_ts)
        if doCUML:
            y_ts=y_ts.to_numpy()
            yhat_ts=yhat_ts.to_numpy()
        return modelResults(y_ts,yhat_ts,'SV')

def doCB(x,y,testBool=None,nTrees=300,cat_features=[],tryGPU=True,paramsIn:dict=None):
    CBDevice = 'GPU' if torch.cuda.is_available() and tryGPU else 'CPU'
    threadCount=threadAllocation(CBDevice)
    if testBool is not None:
        x_ts=x[testBool]
        y_ts=y[testBool]
        x=x[~testBool]
        y=y[~testBool]
    if paramsIn is None:
        cb_classifier = CatBoostClassifier(iterations=nTrees,cat_features=cat_features,thread_count=threadCount,
                                           task_type=CBDevice,verbose=False,random_seed=5723588)
    else:
        cb_classifier = CatBoostClassifier(**paramsIn,cat_features=cat_features,thread_count=threadCount,
                                           task_type=CBDevice,verbose=False,random_seed=5723588)
    cb_classifier.fit(x, y,early_stopping_rounds=35, verbose=300)
    yhat = cb_classifier.predict(x)[:,0]
    if testBool is None:
        return cb_classifier, yhat
    else:
        yhat_ts = cb_classifier.predict(x_ts)[:,0]
        return modelResults(y_ts,yhat_ts,'CB')

class FNN(nn.Module): #inherits the class properties defined in nn.Module
    def __init__(self, input_size, output_size, hidden_sizes,dropout_prob):
        super(FNN, self).__init__() #initiate nn.Module's __init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        if dropout_prob>0:
            self.doDropout=True
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.doDropout=False
    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            if self.doDropout:
                x = self.dropout(x)
        x = self.output_layer(x)
        x = F.softmax(x, dim=1)
        return x

def doNN(x,y,testBool=None,hidden_sizes=[128,256],lrate=0.005,nEpoch=300, dropout_prob=0.2, weight_decay=0.001,plotErrors=False):
    if testBool is not None:
        x_ts=torch.tensor(x[testBool],dtype=torch.float32).to(device)
        y_ts=torch.tensor(y[testBool],dtype=torch.long).to(device)
        x=x[~testBool]
        y=y[~testBool]
        ts_losses=[]
    tr_losses=[]
    input_size=x.shape[1]
    output_size=len(np.unique(y))
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    model = FNN(input_size, output_size, hidden_sizes,dropout_prob).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lrate,weight_decay=weight_decay)
    for epoch in range(nEpoch):
        outputs=model(x)
        loss=criterion(outputs,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_losses.append(loss.item())
        if testBool is not None:
            model.eval()
            with torch.no_grad():
                yhat_ts = model(x_ts)
                ts_loss = criterion(yhat_ts, y_ts)
                ts_losses.append(ts_loss.item())
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{nEpoch}], Train Loss: {loss.item():.4f}, Test Loss: {ts_loss.item():.4f}")
            model.train()
    model.eval()
    if testBool is None:
        with torch.no_grad():
            outputs = model(x)
            _, yhat = torch.max(outputs, 1)
        return model, yhat.cpu().numpy()
    else:
        if plotErrors:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, nEpoch + 1), tr_losses, label="Training Loss")
            plt.plot(range(1, nEpoch + 1), ts_losses, label="Testing Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Hidden Size: {hidden_sizes}, Learning Rate: {lrate:.4f}\nEpochs: {nEpoch}, Weight Decay: {weight_decay:.4f}")
            plt.legend()
            plt.grid()
            plt.show()
        with torch.no_grad():
            outputs = model(x_ts)
            _, yhat = torch.max(outputs, 1)
            return modelResults(y_ts.cpu().numpy(),yhat.cpu().numpy(),'NN')

#%% test
if __name__=='__main__':
    testCatboost=False
    from src.preprocessing import getVars, inputGenerator
    dataIn = pd.read_csv('data/dataIn.csv')
    #round 1
    opts, testBool, y, ylab, dfCombinations = getVars(dataIn)
    if testCatboost: #requires x to be data frame not np.array, needs cat column specification, ...
        x, xlab, currentOpt = inputGenerator(dfRaw=dataIn,opt=opts[0],locOpt='zone',catOpt='catboost')
        cat_features = x.select_dtypes(include=['category', 'object']).columns.tolist()
        doCB(x,y,testBool,cat_features=cat_features)
        modelOut, yhat= doCB(x,y,cat_features=cat_features)
        modelResults(y,yhat,modelName='CB')
    else:
        for row in dfCombinations.iloc[0:1,:].itertuples(): #extract example row
            x, xlab, currentOpt = inputGenerator(dfRaw=dataIn,opt=opts[row.opt],timeOpt=row.timeOpt,locOpt=row.locOpt)
    doXG(x,y,testBool) #whan test index is fed
    modelOut, yhat=doXG(x,y) #when not
    outs=modelResults(y,yhat,modelName='XG')
    #round 2
    opts2, testBool2, y2, ylab2, dfCombinations2 = getVars(dataIn,studyround=2)
    for row2 in dfCombinations2.iloc[827:829,:].itertuples(): #extract example row
        x2, xlab2, currentOpt2 = inputGenerator(dfRaw=dataIn,opt=opts2[row2.opt],timeOpt=row2.timeOpt,locOpt=row2.locOpt,
                                                catOpt=row2.catOpt,encDim=row2.encDim,denom=row2.denom,studyround=2)
        doNN(x2,y2,testBool2)
        modelOut2, yhat2=doNN(x2,y2)
        outs2=modelResults(y2,yhat2,modelName='NN')
elif not _PRINTED_SETUP: #message shown upon import, but only once
    print('mlmodels.py imported')
    _PRINTED_SETUP = True