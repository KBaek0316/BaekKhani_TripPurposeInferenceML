# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18:18:55 2025
@author: baek0040@umn.edu
"""
import os
if '__file__' in globals():
    from pathlib import Path
    WPATH = Path(__file__).resolve().parent
elif os.name=='posix': #linux server; Windows='nt'
    WPATH='/export/scratch/users/baek0040/git/TCML/MLTraining'
else:
    if os.environ['USERPROFILE']==r'C:\Users\baek0040':
        WPATH='C:/Users/baek0040/Documents/GitHub/Trip-Chaining-and-Machine-Learning-Inference/MLTraining'
    else:
        WPATH=os.path.abspath('C:/git/Trip-Chaining-and-Machine-Learning-Inference/MLTraining')
os.chdir(WPATH)

from src.preprocessing import getVars, inputGenerator
from src.mlmodels import doRF, doXG, doSV, doNN, doCB

import time
import numpy as np
import pandas as pd

import optuna
from functools import partial
import gc

STORAGE_NAME='sqlite:///outputs/optunaStudies.db'
NUMTRIALS_INIT=50
NUMTRIALS_FINAL=110
STUDYROUND=2
trial_results=[] #not used in FE; just to avoid errors
dataIn = pd.read_csv('data/dataIn.csv')
opts, testBool, y, ylab, dfCombinations = getVars(dataIn,studyround=STUDYROUND) #,labelColName='typeD' implicit
#%% Feature Engineering Experiment Settings
currentModel=input(f"Type a model to do Optuna studies from {dfCombinations.modelOpt.unique().tolist()} (case-insensitive): ").upper()
if currentModel not in dfCombinations.modelOpt.unique():
    raise Exception('Type valid model name')

dfModel=dfCombinations.loc[dfCombinations.modelOpt==currentModel,:].copy().reset_index(names='globalInd')
print(f'{"Feature engineering options exploration for model "+str(currentModel):*^80}')
try:
    res=pd.read_csv('outputs/'+currentModel+'Tuning.csv',index_col="Index")
    dfModel=dfModel[~dfModel.index.isin(res.index)].copy()
    print(f'Continuing the exploration using MLTraining/{currentModel}Tuning.csv')
except Exception:
    print('Initiating the exploration the first time')

if currentModel=='XG' or currentModel=='CB':
    TRYGPU=(input("Would you like to run this console with the GPU (CUDA) privilege if available? (y/n)").lower()=='y')

if len(dfModel)>0:
    print(f'{len(dfModel)} rows left for {currentModel}')
    BATCH_AMOUNT=int(input("How many rows in dfModel would you like to tune in this console?"))
    startInd = int(input(f"Enter the dfModel Start Index (0 will do from 0 to {BATCH_AMOUNT-1} inclusive): ").strip())
    dfModel=dfModel.loc[startInd:(startInd+BATCH_AMOUNT-1)].copy() #.loc includes the final index
else:
    TRYGPU=True
    print(f'{currentModel} feature engineering has already been explored!')



#%% Tuning Objectives
def optim_score(results):
    if STUDYROUND==1:
        optimScore=(results['accuracy']+results['Home_f1-score']+results['W_f1-score'])/3-0.05*results['zeroPreds']
    else:
        optimScore=results['adj_accuracy'] #(results['Work_f1-score']+results['accuracy'])/2
    return optimScore

def objective_NN(trial,x): #'trial' is Optuna convention; 'x' will be treated later with functools.partial
    start_time = time.time()
    # Define the hyperparameters to tune
    n_layers = trial.suggest_int('n_layers', 1, 3)  # 1 to 3 hidden layers
    hidden_sizes = []
    for i in range(n_layers):
        hidden_sizes.append(trial.suggest_categorical(f'n_units_layer_{i}', [128, 256, 512]))
    lrate = trial.suggest_float('lrate', 1e-3, 1e-1, log=True)
    nEpoch = trial.suggest_int('nEpoch', 200, 500, step=100)
    dropout_prob = trial.suggest_float('dropout_prob', 0.0, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    # Train the model with these hyperparameters
    results = doNN(x, y, testBool=testBool, 
                  hidden_sizes=hidden_sizes,
                  lrate=lrate,
                  nEpoch=nEpoch,
                  dropout_prob=dropout_prob,
                  weight_decay=weight_decay)
    elapsed_time = time.time() - start_time
    trial_results.append(results)
    trial.set_user_attr('WF1',results['Work_f1-score'])
    trial.set_user_attr('ACC',results['accuracy'])
    print(f"Trial {trial.number} finished in {elapsed_time:.2f} seconds (current dfModel row: {row.Index})")
    return optim_score(results)

def objective_RF(trial,x): #'trial' is Optuna convention
    start_time = time.time()
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=50), #ensemble amount
        'max_depth': trial.suggest_categorical('max_depth', list(range(5, 51, 5))), #regularization vs complexity add [200] + for infinite depth
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10), #how aggressive the split can be
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10), # regularization
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']), #feature randomness for generalization
        'max_samples': trial.suggest_float('max_samples', 0.6, 1.0)
    }
    try:
        results = doRF(x,y,testBool=testBool,paramsIn=params)
    except Exception as e:
        if "cudaErrorMemoryAllocation" in str(e):
            raise optuna.exceptions.TrialPruned()
        else:
            raise e
    elapsed_time = time.time() - start_time
    trial_results.append(results)
    trial.set_user_attr('WF1',results['Work_f1-score'])
    trial.set_user_attr('ACC',results['accuracy'])
    print(f"Trial {trial.number} finished in {elapsed_time:.2f} seconds (current dfModel row: {row.Index})")
    return optim_score(results)

def objective_XG(trial,x):
    start_time = time.time()
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 10),  # Controls overfitting
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        #'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # discarded row sampling; ~1 was the best
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 2, log=True),
        'gamma': trial.suggest_float('gamma', 1e-2, 10, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),  # Column sampling
    }
    try:
        results = doXG(x,y,testBool=testBool,paramsIn=params,tryGPU=TRYGPU)
    except Exception as e:
        if "cudaErrorMemoryAllocation" in str(e):
            raise optuna.exceptions.TrialPruned()
        else:
            raise e
    elapsed_time = time.time() - start_time
    trial_results.append(results)
    trial.set_user_attr('WF1',results['Work_f1-score'])
    trial.set_user_attr('ACC',results['accuracy'])
    print(f"Trial {trial.number} finished in {elapsed_time:.2f} seconds (current dfModel row: {row.Index})")
    return optim_score(results)

def objective_SV(trial,x):
    start_time = time.time()
    kernel_choice = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
    params = {
        'C': trial.suggest_float('C', 1e-3, 1e1, log=True),
        'kernel': kernel_choice,
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
    }
    if kernel_choice != 'linear':
        params['gamma'] = trial.suggest_float('gamma', 1e-3, 1, log=True)
        if kernel_choice == 'poly': # Gamma: narrower for poly, wider for rbf, none for linear
            params['degree'] = trial.suggest_int('degree', 2, 4)
            params['coef0'] = trial.suggest_float('coef0', -0.5, 0.5)
    #print(f"[Trial {trial.number}] STARTING with params: {params}")
    results = doSV(x,y,testBool=testBool, paramsIn=params)
    elapsed_time = time.time() - start_time
    trial_results.append(results)
    trial.set_user_attr('WF1',results['Work_f1-score'])
    trial.set_user_attr('ACC',results['accuracy'])
    print(f"Trial {trial.number} finished in {elapsed_time:.2f} seconds (current dfModel row: {row.Index})")
    gc.collect()
    return optim_score(results)

def objective_CB(trial,x):
    start_time = time.time()
    cat_features = x.select_dtypes(include=['category', 'object']).columns.tolist()
    params = {
        'iterations': trial.suggest_int('iterations', 200, 800, step=100),  # More trees = better learning but longer training
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),  # Step size for updates
        'depth': trial.suggest_int('depth', 3, 10),  # Tree depth; higher = more complex models
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10, log=True),  # L2 regularization to prevent overfitting
        'random_strength': trial.suggest_float('random_strength', 0.0, 5.0), #added random perturbation when calc gain or delta_weighted_gini
        #'border_count': trial.suggest_int('border_count', 32, 128),  # Deprecated, more related to computation speed than accuracy
        'auto_class_weights': trial.suggest_categorical('auto_class_weights', [None, 'Balanced', 'SqrtBalanced'])  # For imbalanced datasets
    }
    results = doCB(x,y,cat_features=cat_features,testBool=testBool,paramsIn=params,tryGPU=TRYGPU)
    elapsed_time = time.time() - start_time
    trial_results.append(results)
    trial.set_user_attr('WF1',results['Work_f1-score'])
    trial.set_user_attr('ACC',results['accuracy'])
    print(f"Trial {trial.number} finished in {elapsed_time:.2f} seconds (current dfModel row: {row.Index})")
    return optim_score(results)

# Map model names to objectives; ordered by paper order
objective_functions = {
    "RF": objective_RF,
    "XG": objective_XG,
    "CB": objective_CB,
    "NN": objective_NN,
    "SV": objective_SV
}
#NUMTRIALS_INIT=6
#%% Feature Engineering (FE) Exploration with the given model input
for row in dfModel.itertuples(): #dataset loop: nest this line when finished
    paramsIn={k: v for k, v in row._asdict().items() if k not in ['Index','globalInd','modelOpt']}
    paramsIn['opt']=opts[paramsIn['opt']]
    x, xlab, currentOpt = inputGenerator(dfRaw=dataIn,**paramsIn,studyround=STUDYROUND)
    currentObj = partial(objective_functions[currentModel], x=x) #obtuna only accepts "trial" so have to freeze x
    hpStudy = optuna.create_study(study_name=f'temp_{currentModel}_{row.Index}',storage=STORAGE_NAME,load_if_exists=True,direction='maximize')
    if len(hpStudy.trials)<NUMTRIALS_INIT:
        hpStudy.optimize(currentObj, n_trials=NUMTRIALS_INIT-len(hpStudy.trials))
    trial = hpStudy.best_trial
    print(f"  Value: {trial.value}")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    trials_df = hpStudy.trials_dataframe().dropna(subset=['value']).sort_values("value", ascending=False).reset_index(drop=True)
    trials_df['seconds']=trials_df['duration'].dt.total_seconds().astype(int)
    trials_top5=trials_df.iloc[:5,:].reset_index(drop=True)
    trials_bottom5=trials_df.iloc[-5:,:].reset_index(drop=True)
    print(trials_top5.drop(columns=['number','datetime_start','datetime_complete','duration','state']))
    currentRes=pd.DataFrame.from_dict({**row._asdict(), **trial.params},orient='index').transpose()
    currentRes.set_index('Index',inplace=True)
    currentRes['objmean']=trials_df['value'].mean()
    currentRes['objsd']=trials_df['value'].std()
    currentRes['meantop5']=trials_top5['value'].mean()
    currentRes['sdtop5']=trials_top5['value'].std()
    currentRes['meanbottom5']=trials_bottom5['value'].mean()
    currentRes['sdbottom5']=trials_bottom5['value'].std()
    currentRes['meantime']=trials_df['seconds'].mean()
    currentRes['long5time']=trials_df['seconds'].nlargest(5).mean()
    currentRes['bestobj']=trial.value
    currentRes['wf1top5']=trials_df['user_attrs_WF1'].nlargest(5).mean()
    currentRes['acctop5']=trials_df['user_attrs_ACC'].nlargest(5).mean()
    try: # do this again to handle potential in-between updates made by a different console run
        res=pd.read_csv('outputs/'+currentModel+'Tuning.csv',index_col="Index")
        res=pd.concat([res,currentRes])
        res.to_csv('outputs/'+currentModel+'Tuning.csv',index=True)
    except:
        currentRes.to_csv('outputs/'+currentModel+'Tuning.csv',index=True)
    optuna.delete_study(study_name=f'temp_{currentModel}_{row.Index}',storage=STORAGE_NAME)
    trial_results=[] # not for exploration, for actual study below
    print(f'{"Pausing for a cooldown":-^80}')
    time.sleep(5)

#%% Analyze Best and Worst FE scenarios; iterating all models (to be run only in IDE like Spyder)
#NUMTRIALS_FINAL=10
if '__file__' in globals():
    raise Exception(f'{currentModel} exploration has been completed; proceed with an IDE env for final HP analyses')
print([summary.study_name for summary in optuna.get_all_study_summaries(storage=STORAGE_NAME)])
freshStart=False
#main loop
for currentModel in objective_functions.keys(): #['SV','RF','XG','CB','NN'] or objective_functions.keys()
    print(f'{"Hyperparameter exploration for model "+str(currentModel):*^80}')
    dfFE=pd.read_csv('outputs/'+currentModel+'Tuning.csv',index_col="Index")
    optcols=np.intersect1d(dfFE.columns,['opt','timeOpt','locOpt','catOpt','encDim','denom'])
    for option in ['best','worst']: #['best','worst']
        trial_results=[]
        if freshStart:
            try:
                optuna.delete_study(study_name='final_'+currentModel+option, storage=STORAGE_NAME)
            except KeyError:
                print(f"Final round for {currentModel}")
        row=pd.Series({'name':'Fake Row for Reporting', 'Index': option+'_'+currentModel})
        if option=='best':
            currentRow=dfFE.loc[dfFE['bestobj'].idxmax(),optcols].to_dict()
        else:
            currentRow=dfFE.loc[dfFE['bestobj'].idxmin(),optcols].to_dict()
        currentRow['opt']=opts[currentRow['opt']]
        x, xlab, currentOpt = inputGenerator(dfRaw=dataIn,**currentRow,studyround=STUDYROUND)
        finalObj = partial(objective_functions[currentModel], x=x) # needed to reload changed x
        finalStudy = optuna.create_study(study_name='final_'+currentModel+option,storage=STORAGE_NAME,load_if_exists=True, direction='maximize')
        finalStudy.optimize(finalObj, n_trials=(NUMTRIALS_FINAL-len(finalStudy.trials)))
        finalResults = finalStudy.trials_dataframe().drop(columns=['number','datetime_start','datetime_complete','state'])
        finalResults['duration']=finalResults['duration'].dt.total_seconds().astype(int)
        finalResults = pd.concat([finalResults,pd.DataFrame(trial_results)],axis=1).sort_values("value", ascending=False).reset_index(drop=True)
        finalResults['model']=currentModel
        finalResults['option']=option
        finalResults.to_csv('outputs/Final_'+currentModel+option+'.csv',index=False)

