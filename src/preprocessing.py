# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 20:55:49 2025

"""

import os
import numpy as np
import pandas as pd
from haversine import haversine
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import random
from itertools import product

random.seed(5723588)
np.random.seed(5723588)

_PRINTED_SETUP = False
LABELNAME='typeD'
OPTDICT = {
    'month': {
        '4seasons': {1: 3, 2: 3, 3: 3, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 2, 10: 2, 11: 2, 12: 3},
        '2seasons': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 0, 10: 0, 11: 0, 12: 0}
    },
    'dow': {
        '5level': {'Monday': 0, 'Tuesday': 1, 'Wednesday': 1, 'Thursday': 1, 'Friday': 2, 'Saturday': 3, 'Sunday': 4},
        '4level': {'Monday': 0, 'Tuesday': 1, 'Wednesday': 1, 'Thursday': 1, 'Friday': 2, 'Saturday': 3, 'Sunday': 3},
        '3level': {'Monday': 0, 'Tuesday': 0, 'Wednesday': 0, 'Thursday': 0, 'Friday': 1, 'Saturday': 2, 'Sunday': 2},
        '2level': {'Monday': 0, 'Tuesday': 0, 'Wednesday': 0, 'Thursday': 0, 'Friday': 0, 'Saturday': 1, 'Sunday': 1}
    },
    'nTrans': {
        'absolute': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
        'binary': {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
    }
}
TIMEOPTS = ['raw', 'addDur', 'cyclic', '1dpos']
LOCOPTS = ['raw', 'zone', '2dpos', 'polar', 'elliptic']
CATOPTS = ['onehot','WTE','5FTE']
MODELOPTS = ['SV','RF','XG','CB','NN'] #order: ['SV','RF','XG','CB','NN']

if __name__=='__main__':
    if '__file__' in globals():
        from pathlib import Path
        WPATH = Path(__file__).resolve().parent.parent #.parent=src .parent.parent=MLTraining
    elif os.name=='posix': #linux server
        WPATH='/export/scratch/users/baek0040/git/TCML/MLTraining'
    else:
        if os.environ['USERPROFILE']==r'C:\Users\baek0040':
            WPATH='C:/Users/baek0040/Documents/GitHub/Trip-Chaining-and-Machine-Learning-Inference/MLTraining'
        else:
            WPATH=os.path.abspath('C:/git/Trip-Chaining-and-Machine-Learning-Inference/MLTraining')
    os.chdir(WPATH)

'''for debugging
dataIn = pd.read_csv(r'data/dataIn.csv')
dfRaw, optDict, timeOpts, locOpts, catOpts, modelOpts, labelColName, studyround= (dataIn, OPTDICT,TIMEOPTS, LOCOPTS ,CATOPTS,MODELOPTS, LABELNAME, 2)
'''

def getVars(dfRaw,optDict:dict=OPTDICT,timeOpts:list=TIMEOPTS,locOpts: list = LOCOPTS,
            catOpts: list = CATOPTS,modelOpts: list = MODELOPTS,labelColName=LABELNAME, studyround=1):
    dfRaw=dfRaw.copy()
    if optDict is None:
        raise ValueError("OPTDICT must be provided! visit preprocessing.py for the default OPTDICT")
    if studyround==2:
        newlabel=np.full(len(dfRaw),'Others')
        newlabel[dfRaw[LABELNAME]=='Home']='Home'
        newlabel[dfRaw[LABELNAME]=='W']='Work'
        newlabel[dfRaw[LABELNAME]=='Sh']='Shop'
        #newlabel[dfRaw[LABELNAME].isin(['Sh','So'])]='Leis'
        dfRaw[LABELNAME]=newlabel
    _, testInd = train_test_split(dfRaw.index, stratify=dfRaw[labelColName])
    y, ylab = pd.factorize(dfRaw[LABELNAME])
    testBool=np.isin(np.arange(len(y)), testInd)
    opts = [dict(zip(optDict.keys(), combination)) for combination in product(*[list(d.keys()) for d in optDict.values()])]
    if studyround==2:
        opts=[opts[0], opts[-1]]
    dfCombinations=list(product(list(range(len(opts))),timeOpts,locOpts,catOpts,modelOpts))
    dfCombinations=pd.DataFrame(dfCombinations, columns=['opt', 'timeOpt', 'locOpt','catOpt','modelOpt'])
    if studyround==2:
        dfCombinations=dfCombinations.loc[dfCombinations.timeOpt!='addDur',:]
        encdf=pd.DataFrame(list(product([8,16,24], [2500,5000,10000])), columns=['encDim', 'denom'])
        dfCombinationsPE=pd.merge(dfCombinations.loc[dfCombinations.locOpt=='2dpos',:],encdf,how='cross')
        dfCombinations=pd.concat([dfCombinations.loc[dfCombinations.locOpt!='2dpos',:],dfCombinationsPE],ignore_index=True)
        dfCombinations[['encDim', 'denom']] = dfCombinations[['encDim', 'denom']].astype('Int64') #Int64 can handle NA
        #dfCombinations=dfCombinations.loc[~((dfCombinations.opt==(len(opts)-1)) & (dfCombinations.catOpt!='onehot')),:] #all binary
    else:
        dfCombinations=dfCombinations.loc[dfCombinations.catOpt=='onehot',:]
    dfCombinations.loc[dfCombinations.modelOpt=='CB','catOpt']='catboost'
    dfCombinations=dfCombinations.drop_duplicates() #remove CB's duplicated rows
    sorter_index = {v: i for i, v in enumerate(MODELOPTS)}
    dfCombinations['sort_key'] = dfCombinations['modelOpt'].map(sorter_index)
    sort_cols=np.concatenate((['sort_key', 'opt', 'timeOpt', 'locOpt', 'catOpt'],
                              dfCombinations.columns[dfCombinations.columns.isin(['encDim', 'denom'])].values))
    dfCombinations=dfCombinations.sort_values(by=sort_cols.tolist(),ignore_index=True).drop(columns='sort_key')
    return opts, testBool, y, ylab, dfCombinations

#%% Defining input converters
def marginalDist(centerlat = 44.955, centerlon = -93.102):
    xmar = 1000 * haversine((centerlat, centerlon), (centerlat, centerlon + 0.001), unit='mi')
    ymar = 1000 * haversine((centerlat, centerlon), (centerlat + 0.001, centerlon), unit='mi')
    return xmar, ymar

# Convert inputs for uniform processing
def to_iterable(x):
    if isinstance(x, (int, float)):
        return np.array([x])
    elif isinstance(x, (pd.Series,np.ndarray)):
        return x
    else:
        raise ValueError("Input must be a scalar, np.array, or pd.Series.")

def cyclic(raw,colName:str=None)->pd.DataFrame:
    raw=to_iterable(raw)
    sin_t = np.sin(2 * np.pi * raw / 1440)
    cos_t = np.cos(2 * np.pi * raw / 1440)
    # Create DataFrame
    if colName is None:
        colName = "tri" if not isinstance(raw, pd.Series) else raw.name
    columns = [colName+'_sin',colName+'_cos']
    df=pd.DataFrame(np.column_stack([sin_t, cos_t]),columns=columns)
    return df

def pos_1d(raw, outDim=4, denom=250, colName:str=None)->pd.DataFrame:
    raw=to_iterable(raw) #denomn=250 can handle 1570 pos
    outDim=int(outDim)
    denom=int(denom)
    # Initialize output array
    pe = np.zeros((len(raw), outDim))
    # Compute positional encoding
    for i in range(0, outDim, 2):
        div_term = np.exp(i * -np.log(denom) / outDim)
        pe[:, i] = np.sin(raw * outDim)
        if i + 1 < outDim:
            pe[:, i + 1] = np.cos(raw * div_term)
    # Create DataFrame
    if colName is None:
        colName = "pos" if not isinstance(raw, pd.Series) else raw.name
    columns = [f"{colName}_{i+1}" for i in range(outDim)]
    df = pd.DataFrame(pe, columns=columns)
    return df

def pos_2d(lat, lon, outDim=4, denom=100, colName:str=None,scaled=False)->pd.DataFrame:
    #ensure the lowest frequency (longest wavelength) is not shorter than your data’s span
    lat = to_iterable(lat)
    lon = to_iterable(lon)
    outDim=int(outDim)
    denom=int(denom)
    if scaled:
        lat=(lat-lat.min())*10000 #max diff~1.3 degrees->13117.68 positions
        lon=(lon-lon.min())*10000 #denom 10000 can handle 62831 pos
    # Compute positional encoding for latitude
    pe = np.zeros((len(lat), outDim * 2))  # init
    for i in range(0, outDim, 2):
        div_term = np.exp(i * -np.log(denom) / outDim)
        pe[:, i] = np.sin(lat * div_term)
        pe[:, outDim + i] = np.sin(lon * div_term)
        if i + 1 < outDim:
            pe[:, i + 1] = np.cos(lat * div_term)
            pe[:, outDim + i + 1] = np.cos(lon * div_term)
    # Create DataFrame
    if colName is None:
        colname_lat = "lat" if not isinstance(lat, pd.Series) else lat.name
        colname_lon = "lon" if not isinstance(lon, pd.Series) else lon.name
    else:
        colname_lat = f"{colName}_lat"
        colname_lon = f"{colName}_lon"
    columns = (
        [f"{colname_lat}_{i+1}" for i in range(outDim)] +
        [f"{colname_lon}_{i+1}" for i in range(outDim)]
    )
    df = pd.DataFrame(pe, columns=columns)
    return df

def polar(lat, lon, prefix='',center_lat=44.95893, center_lon=-93.22423,rangeTwoPi=False):
    XMAR, YMAR = marginalDist()
    lat = to_iterable(lat)
    lon = to_iterable(lon)
    def latlon_distance(lat1, lon1, lat2, lon2):
        return ((XMAR*(lon1-lon2))**2 + (YMAR*(lat1-lat2))**2)**0.5
    def compute_angle(latitude, longitude):#Compute the angle theta in radians relative to the center
        delta_lon = np.radians(XMAR*(longitude - center_lon))
        delta_lat = np.radians(YMAR*(latitude - center_lat))
        return np.arctan2(delta_lat, delta_lon)
    df=pd.DataFrame({'lat':lat,'lon':lon})
    df[prefix+"_r"] = df.apply(lambda row: latlon_distance(row['lat'], row['lon'], center_lat, center_lon), axis=1)
    df[prefix+"_theta"] = df.apply(lambda row: compute_angle(row['lat'], row['lon']), axis=1)
    if rangeTwoPi:
        df[prefix+"_theta"] = np.where(df[prefix+"_theta"] < 0, df[prefix+"_theta"] + 2 * np.pi, df[prefix+"_theta"])
    df=df.drop(columns=['lat','lon'])
    return df

def elliptic(lat_col, lon_col, prefix='', f1_lat=44.97731, f1_lon=-93.26561, f2_lat=44.94426, f2_lon=-93.09407,rangeTwoPi=False):
    XMAR, YMAR = marginalDist()
    # Convert lat/lon to Cartesian (miles) with scaling
    x = (lon_col - f1_lon) * XMAR
    y = (lat_col - f1_lat) * YMAR
    # Convert foci to Cartesian
    x1, y1 = 0, 0  # First focus (temporary anchor point)
    x2 = (f2_lon - f1_lon) * XMAR
    y2 = (f2_lat - f1_lat) * YMAR
    # Midpoint of the foci (center of ellipse)
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
    # Translate points so that midpoint is at (0,0)
    x -= xc
    y -= yc
    x1 -= xc
    y1 -= yc
    x2 -= xc
    y2 -= yc
    # Compute rotation angle (align major axis with x-axis)
    theta = np.arctan2(y2 - y1, x2 - x1)
    # Rotate all points
    x_rot = x * np.cos(-theta) - y * np.sin(-theta)
    y_rot = x * np.sin(-theta) + y * np.cos(-theta)
    x1_rot = x1 * np.cos(-theta) - y1 * np.sin(-theta)
    y1_rot = x1 * np.sin(-theta) + y1 * np.cos(-theta)
    x2_rot = x2 * np.cos(-theta) - y2 * np.sin(-theta)
    y2_rot = x2 * np.sin(-theta) + y2 * np.cos(-theta)
    # Compute semi-major axis (a)
    a = np.linalg.norm([x2_rot - x1_rot, y2_rot - y1_rot]) / 2
    # Compute elliptic coordinates (μ, ν)
    r1 = np.sqrt((x_rot - x1_rot) ** 2 + (y_rot - y1_rot) ** 2)
    r2 = np.sqrt((x_rot - x2_rot) ** 2 + (y_rot - y2_rot) ** 2)
    u = np.arccosh((r1 + r2) / (2 * a))
    v = np.arctan2(y_rot, x_rot)
    df=pd.DataFrame({prefix+'_u':u,prefix+'_v':v})
    if rangeTwoPi:
        df[prefix+"_v"] = np.where(df[prefix+"_v"] < 0, df[prefix+"_v"] + 2 * np.pi, df[prefix+"_v"])
    print(f'The length of semi-major axis a is {a:.4f} miles')
    return df

def coordplots(df,c_system='Cartesian',labs=None,borders=None):
    '''
    df=dataIn.copy()
    labs=ylab2
    c_system='Elliptic'
    borders='borders.geojson'
    labs={'Sh':'Shopping','O':'Others','So':'Others','Ed':'Others','Home':'Home','W':'Work'}
    '''
    import matplotlib.pyplot as plt
    import geopandas as gpd
    fig, ax = plt.subplots(figsize=(4,5))
    if labs is not None:
        df['DestinationType'] = df['typeD'].map(labs)
    if c_system == 'Cartesian':
        plt.title('Earth Coordinates')
        dfOri=df[['OriLon','OriLat',]].copy()
        dfDes=df[['DesLon','DesLat']].copy()
        colnames=['Longitude','Latitude']
        ax.set_title('Earth Coordinates')
        if borders is not None:
            border = (gpd.read_file(f"data/{borders}").to_crs(epsg=4326))
            border.plot(ax=ax,facecolor="none",edgecolor="black",linewidth=2)
            ax.set_aspect('auto')
        ax.set_xlim(-93.6, -92.9)
        ax.set_ylim(44.7, 45.3)
    else:
        plt.title(c_system)
        import matplotlib.ticker as ticker
        def multiple_of_pi_formatter(val, pos):
            fractions = {-1*np.pi: "-π",-3*np.pi/4: "-3/4π",-1*np.pi/2: "-1/2π",
                         -1*np.pi/4: "-1/4π",0: "0", np.pi/4: "1/4π",
                         np.pi/2: "1/2π", 3*np.pi/4: "3/4π", np.pi: "π"}
            for key in fractions.keys():
                if np.isclose(val, key, atol=1e-2):  # Handle float precision
                    return fractions[key]
            return ""
        ax.yaxis.set_major_locator(ticker.FixedLocator([-np.pi, -3*np.pi/4, -1*np.pi/2, -1*np.pi/4,
                                                        0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(multiple_of_pi_formatter))
    if c_system == 'Polar':
        dfOri=polar(df.OriLat,df.OriLon)
        dfDes=polar(df.DesLat,df.DesLon)
        colnames=['radius_mile','theta']
        ax.set_xlim(0,20)
    if c_system=='Elliptic':
        dfOri=elliptic(df.OriLat,df.OriLon)
        dfDes=elliptic(df.DesLat,df.DesLon)
        colnames=['u (On an ellipse with equal focal distance sum)','v (Angular position from an intersecting hyperbolae)']
        ax.set_xlim(0,2.5)
    ax.set_xlabel(colnames[0])
    ax.set_ylabel(colnames[1])
    if labs is None:
        dfConcat=pd.concat([dfOri,dfDes])
        plt.scatter(dfConcat[colnames[0]],dfConcat[colnames[1]],c=dfConcat['color'],s=10,alpha=0.3)
        if c_system == 'Cartesian':
            ax.legend(title="Location Type", loc="best")
    else:
        dfConcat=dfDes.copy()
        dfConcat.columns=colnames
        dfConcat['DestinationType']=df['DestinationType']
        cmap = plt.get_cmap("tab10")
        categories = list(dict.fromkeys(labs.values()))
        for i, category in enumerate(categories):
            subset = dfConcat[dfConcat["DestinationType"] == category]
            ax.scatter(subset[colnames[0]], subset[colnames[1]],label=category,color=cmap(i),s=6,alpha=0.15,zorder=10 + i)
        if c_system == 'Cartesian':
            ax.legend(title="Destination\nPlace Type", loc="best")
    plt.tight_layout()
    plt.show()
    return None

'''for debugging
dataIn = pd.read_csv(r'data/dataIn.csv')
opts, testBool, y, ylab, dfCombinations = getVars(dataIn,OPTDICT,studyround=2)
dfRaw, opt, timeOpt, locOpt, catOpt, encDim, denom, studyround= (dataIn, opts[0],'1dpos', '2dpos' ,'5FTE', 4, 1000, 2)
'''

def inputGenerator(dfRaw,opt,timeOpt='raw',locOpt='2dpos',catOpt='onehot',encDim=4,denom=1000,studyround=1):
    opt=opt.copy()
    dfConverted=dfRaw.copy()
    if studyround==2:
        newlabel=np.full(len(dfConverted),'Others')
        newlabel[dfConverted[LABELNAME]=='Home']='Home'
        newlabel[dfConverted[LABELNAME]=='W']='Work'
        newlabel[dfConverted[LABELNAME]=='Sh']='Shop'
        #newlabel[dfConverted[LABELNAME].isin(['Sh','So'])]='Leis'
        dfConverted[LABELNAME]=newlabel
    if studyround!=3: #when applying trained models to AFC
        label, lab= pd.factorize(dfConverted[LABELNAME])
        label_series = pd.Series(label)
    else:
        label, lab=0,0
    scaler=StandardScaler()
    addedCols=[]
    delCols=['sid','aid','routes',LABELNAME,'tr']
    stdCols=['nwDist','eucDist']
    catCols=list(opt.keys())
    alpha=5
    for key, value in opt.items():
        dfConverted[key]=dfConverted[key].map(OPTDICT[key][value])
    for col in catCols.copy():
        if dfConverted[col].nunique()==2: #if binary variable
            catCols.remove(col) #do not target encode
    print(f'{"Input Dataset Generation":-^80}')
    print(f'1.Using the following conversions for the selected categorical variables:\n {opt}')
    match timeOpt:
        case 'raw':
            print('2. Using standardized initial boarding and final alighting timestamps')
            stdCols.extend(['dep','arr'])
        case 'addDur':
            print('2. Using raw timestamps and travel time duration')
            dfConverted['duration']=dfConverted.arr-dfConverted.dep
            stdCols.extend(['dep','arr','duration'])
        case 'cyclic':
            print('2. Using trigonometric function-converted timestamps')
            dfConverted=pd.concat([dfConverted,cyclic(dfRaw.dep)],axis=1)
            dfConverted=pd.concat([dfConverted,cyclic(dfRaw.arr)],axis=1)
            addedCols.extend(['dep_sin','dep_cos','arr_sin','arr_cos'])
            if studyround==1: #replace raw timestamps
                delCols.extend(['dep','arr'])
            else: #in addition to raw timestamps
                stdCols.extend(['dep','arr'])
        case '1dpos':
            dimfixed1d=4 #originally this block's dimfixed1d was encDim
            print('2. Using 1D-positional encoding for the timestamps with output dimension 4')
            dfConverted=pd.concat([dfConverted,pos_1d(dfRaw.dep,dimfixed1d)],axis=1)
            dfConverted=pd.concat([dfConverted,pos_1d(dfRaw.arr,dimfixed1d)],axis=1)
            addedCols.extend(([f'dep_{i+1}' for i in range(dimfixed1d)]+[f'arr_{i+1}' for i in range(dimfixed1d)]))
            if studyround==1:
                delCols.extend(['dep','arr'])
            else:
                stdCols.extend(['dep','arr'])
        case 'cyclicdDur':
            pass
    match locOpt:
        case 'raw':
            print('3. Using standardized origin and destination lat/lon coordinates')
            stdCols.extend(['OriLat','OriLon','DesLat','DesLon'])
            delCols.extend(['OriZone','DesZone'])
        case 'zone':
            print('3. Using zone membership defined by the MetCouncil Transit Market Area and Admin Borders')
            catCols.extend(['OriZone','DesZone'])
            if studyround==1:
                delCols.extend(['OriLat','OriLon','DesLat','DesLon'])
            else:
                stdCols.extend(['OriLat','OriLon','DesLat','DesLon'])
        case '2dpos':
            print(f'3. Using 2D-positional encoding for the locations with dimension {encDim}, denominator {denom}')
            encDim=int(encDim)
            denom=int(denom)
            pescale=True if studyround>1 else False
            dfConverted=pd.concat([dfConverted,pos_2d(dfRaw.OriLat,dfRaw.OriLon,encDim,scaled=pescale)],axis=1)
            dfConverted=pd.concat([dfConverted,pos_2d(dfRaw.DesLat,dfRaw.DesLon,encDim,scaled=pescale)],axis=1)
            addedCols.extend(([f'OriLat_{i+1}' for i in range(encDim)]+[f'OriLon_{i+1}' for i in range(encDim)]))
            addedCols.extend(([f'DesLat_{i+1}' for i in range(encDim)]+[f'DesLon_{i+1}' for i in range(encDim)]))
            delCols.extend(['OriZone','DesZone'])
            if studyround==1:
                delCols.extend(['OriLat','OriLon','DesLat','DesLon'])
            else:
                stdCols.extend(['OriLat','OriLon','DesLat','DesLon'])
        case 'polar':
            print('3. Using polar-converted coordinate system for the locations')
            dfConverted=pd.concat([dfConverted,polar(dfRaw.OriLat,dfRaw.OriLon,'ori')],axis=1)
            dfConverted=pd.concat([dfConverted,polar(dfRaw.DesLat,dfRaw.DesLon,'des')],axis=1)
            addedCols.extend(['ori_r','ori_theta','des_r','des_theta'])
            stdCols.extend(['ori_r','des_r']) #['ori_r','ori_theta','des_r','des_theta']
            delCols.extend(['OriZone','DesZone'])
            if studyround==1:
                delCols.extend(['OriLat','OriLon','DesLat','DesLon'])
            else:
                stdCols.extend(['OriLat','OriLon','DesLat','DesLon'])
        case 'elliptic':
            print('3. Using elliptic-converted coordinate system for the locations')
            dfConverted=pd.concat([dfConverted,elliptic(dfRaw.OriLat,dfRaw.OriLon,'ori')],axis=1)
            dfConverted=pd.concat([dfConverted,elliptic(dfRaw.DesLat,dfRaw.DesLon,'des')],axis=1)
            addedCols.extend(['ori_u','ori_v','des_u','des_v'])
            stdCols.extend(['ori_u','des_u']) #['ori_u','ori_v','des_u','des_v']
            delCols.extend(['OriZone','DesZone'])
            if studyround==1:
                delCols.extend(['OriLat','OriLon','DesLat','DesLon'])
            else:
                stdCols.extend(['OriLat','OriLon','DesLat','DesLon'])
    match catOpt:
        case 'onehot':
            print('4. Using the one-hot encoder for the final categorical variable conversion')
            catEnc=OneHotEncoder(sparse_output=False)
            dfOnehot=pd.DataFrame(catEnc.fit_transform(dfConverted[catCols]),columns=catEnc.get_feature_names_out())
            dfConverted=pd.concat([dfConverted,dfOnehot],axis=1)
            delCols.extend(catCols)
            addedCols.extend(catEnc.get_feature_names_out().tolist())
        case 'WTE':
            print('4. Using the target encoder for the final categorical variable conversion (alpha=5)')
            global_means = label_series.value_counts(normalize=True).sort_index()
            global_means.index = lab
            for col in catCols:
                stats = dfConverted.groupby([col, LABELNAME]).size().unstack(fill_value=0)
                counts = stats.sum(axis=1)
                for target_label in lab:
                    class_counts = stats.get(target_label, pd.Series(0, index=stats.index)).fillna(0)
                    smoothed = (class_counts + alpha * global_means[target_label]) / (counts + alpha)
                    new_col = f'{col}_WTE_{target_label}'
                    dfConverted[new_col] = dfConverted[col].map(smoothed)
                    addedCols.append(new_col)
            delCols.extend(catCols)
        case '5FTE':
            print('4. Using the 5-fold target encoder for the final categorical variable conversion')
            kf = KFold(n_splits=5, shuffle=True, random_state=5723588)
            oof_encoded = pd.DataFrame(index=dfConverted.index) #working dataframe
            for col in catCols:
                stats_cols = dfConverted[[col, LABELNAME]].copy()
                for target_label in lab:
                    oof_feature = np.zeros(len(dfConverted)) #working col
                    class_idx={lab[i]: i for i in range(len(lab))}[target_label]
                    for train_idx, valid_idx in kf.split(dfConverted):
                        train_data = stats_cols.iloc[train_idx]
                        valid_data = stats_cols.iloc[valid_idx]
                        train_y = label_series.iloc[train_idx]
                        fold_global_means = train_y.value_counts(normalize=True).sort_index()
                        global_mean = fold_global_means.loc[class_idx] if class_idx in fold_global_means.index else 0
                        stats = train_data.groupby(col)[LABELNAME].value_counts().unstack(fill_value=0)
                        counts = stats.sum(axis=1)
                        class_counts = stats.get(target_label, pd.Series(0, index=counts.index)).fillna(0)
                        smoothed = (class_counts + alpha * global_mean) / (counts + alpha)
                        oof_feature[valid_idx] = valid_data[col].map(smoothed).fillna(global_mean).values
                    new_col = f'{col}_5FT_{target_label}'
                    oof_encoded[new_col] = oof_feature
                    addedCols.append(new_col)
            dfConverted = pd.concat([dfConverted, oof_encoded], axis=1)
            delCols.extend(catCols)
        case 'LOOTE':
            print('4. Using the Leave-one-out target encoder for the final categorical variable conversion')
            pass
        case 'catboost': #i.e., apply ordered target encoding; requires a dataframe output not numpy array
            dfConverted[catCols] = dfConverted[catCols].astype('category')
            print('4. Leave the categorical variables as-is for catboost classifier')
    opt.update({'time':timeOpt,'location':locOpt,'catEncoding':catOpt})
    dfConverted[stdCols] = scaler.fit_transform(dfConverted[stdCols])
    dfConverted=dfConverted.drop(columns=delCols,errors='ignore')
    labels=dfConverted.columns
    if catOpt!='catboost': #catboost needs dataframe structure to distinguish cate columns
        dfConverted=dfConverted.to_numpy()
    print(f'5. {len(addedCols)} new columns added')
    print(f'6. The following columns were standardized: {stdCols}')
    print(f'7. Columns deleted from the raw data: {delCols}')
    print(f'{"Input Shape: "+str(dfConverted.shape):-^80}')
    return dfConverted, labels, opt


#%% test
if __name__=='__main__':
    dataIn = pd.read_csv(r'data/dataIn.csv')
    print(dataIn[LABELNAME].value_counts())
    cyclic(dataIn.dep)
    pos_1d(dataIn.arr,4)
    pos_2d(dataIn.OriLat, dataIn.OriLon, outDim=4,denom=10000, colName="ori",scaled=True)
    polar(dataIn.OriLat,dataIn.OriLon,'ori')
    elliptic(dataIn.DesLat,dataIn.DesLon,'des')
    #round 1
    opts, testBool, y, ylab, dfCombinations = getVars(dataIn,OPTDICT)
    dfCombinations.to_csv(r'data/study1keys.csv',index_label='globalInd') #order: ['SV','RF','XG','CB','NN']
    dfConverted, xlab, opt=inputGenerator(dataIn,opts[0],'raw','zone','catboost')
    #round 2
    opts2, testBool2, y2, ylab2, dfCombinations2 = getVars(dataIn,OPTDICT,studyround=2)
    dfCombinations2.to_csv(r'data/study2keys.csv',index_label='globalInd') #order: ['SV','RF','XG','CB','NN']
    dfConverted2, xlab2, opt2=inputGenerator(dataIn,opts2[0],'raw','2dpos','WTE',6,10000,studyround=2) #WTE 5FTE
    #plots
    coordplots(dataIn,'Cartesian')
    coordplots(dataIn,'Polar')
    coordplots(dataIn,'Elliptic')
elif not _PRINTED_SETUP: #message shown upon import, but only once
    print('preprocessing.py imported; Setup:')
    print(f'  (1) Dependent variable or label column is set to: {LABELNAME}')
    print(f'  (2) Categorical variables whose levels are to be consolidated: {list(OPTDICT.keys())}')
    print(f'  (3) Defined feature engineering for the time variables (dep and arr): {TIMEOPTS}')
    print(f'  (4) Defined feature engineering for the location variables (Ori/Des Lat/Lon): {LOCOPTS}')
    print(f'  (5) Defined feature encoding in relations with the label: {CATOPTS}')
    print(f'  (6) Defined machine learning models in mlmodels.py: {MODELOPTS}')
    _PRINTED_SETUP = True

''' index reorganization
dfMother=dfCombinations2.reset_index(names='globalInd')
currentModel='RF'
dfModel=dfMother.loc[dfMother.modelOpt==currentModel,:].copy().reset_index(drop=True).reset_index(names='localInd')
dfAlready=pd.read_csv('outputs/'+currentModel+'Tuning.csv',index_col="Index").sort_index().reset_index(names='localInd')
dfKey=pd.merge(dfAlready,dfModel,how='left',on=['opt','timeOpt','locOpt','catOpt','modelOpt','encDim','denom'])
dfKey=dfKey.rename(columns={'globalInd_y':'globalInd','localInd_y':'localInd'}).drop(columns=['localInd_x','globalInd_x'])
dfKey=dfKey.loc[:,dfAlready.columns].rename(columns={'localInd':'Index'})
dfKey.to_csv('outputs/'+currentModel+'Tuning.csv',index=False)
'''

