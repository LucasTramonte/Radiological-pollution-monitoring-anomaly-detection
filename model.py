
import os
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM ,Dropout , Dense 
import pandas as pd 
import numpy as np 
from scipy.stats import norm
#from sklearn.preprocessing import MinMaxScaler

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# create the LSTM model 


def create_model(n_0 ,n_1 ,input=1):

    model=Sequential([LSTM(n_0, input_shape=(None,input),  return_sequences=True),
                      Dropout(0.2),
                      LSTM(n_1) ,
                      Dropout(0.2),
                      Dense(1)])
    return model 

def train_model (model ,X_train,Y_train,n_epoch):
    
    model.compile(loss='mse',optimizer='Adam')

    model.fit(X_train , Y_train, epochs=n_epoch , validation_split=0.1 )

    return model 

# prepare the training data

def create_sequence(seq ,ls ,nseq):

    X=[]
    y=[]

    for i in range (nseq):
        X.append(seq[i:i+ls])
        y.append(seq[i+ls])

    return np.array(X) ,np.array(y)

df=pd.read_csv("Radiological-pollution-monitoring-anomaly-detection/Assets/Data/2015_months_DebitDoseA.csv")

df_train=df.iloc[0:2501]['24/02/2015 11:20']
X_train = df_train.to_numpy()


X ,Y = create_sequence(X_train,100,2400)
 
# threshold determination   : The first method consist of making the assumption that the errors are gaussian and calculate threshold 


def classify( x_test ,model ,thres, x_val , y_val):

    errors = np.abs(model.predict(x_val)-y_val)

    mu=np.mean(errors)
    sigma= np.std(errors)

    p = [norm.pdf(x, mu , sigma) for x in x_test]

    prediction =np.array([ 1 if p> thres else 0 ])

    return prediction


# second thresholding method 
def continuous_seq( thres , es ):

    ind_a =[ i for i in range(0,len(es)) if es[i]> thres  ]
    ls=[]
    l=[]
    for i in range (0 ,len(ind_a)-1):

        if  ind_a[i]+1==ind_a[i+1]:
            
            ind_a[i+1]=ind_a[i+1]-1

    return ind_a

l= [1,2,3,0.1, 4.5,7,6,9,3,1,1.1]
print(continuous_seq(2 ,l))

def dynamic_tresholding ( eh , x_test, y_test , model ):

    Z=[2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
    
    e=abs(model.predict(x_test)-y_test)

    es=eh+[e]
    es=np.array(es)

    #  Exponential  weighted  moving  average  of errors 
    es=np.exp(es)/np.sum(np.exp(es))
    
    mu,sigma=np.mean(es),np.std(es)

    eps=[ mu+z*sigma for z in Z ]

    normal=[[es < ep] for ep in eps]

    delta_mu ,delta_sigma=[np.mean(es) - np.mean(seq) for seq in normal],[np.std(es) - np.std(seq) for seq in normal]
    
    criteria= []

    pass
    



    




 





