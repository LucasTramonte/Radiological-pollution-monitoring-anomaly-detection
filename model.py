
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

n_neurons=[120,80]


model=Sequential([LSTM(120,input_shape=(None,1),  return_sequences=True),
                  Dropout(0.2),
                  LSTM(80) ,
                  Dropout(0.2),
                  Dense(1)])


model.summary()


model.compile(loss='mse',optimizer='Adam')

#model.fit(X_train,y_train,epochs=10,validation_split=0.1)

# prepare the training data

df=pd.read_csv("Radiological-pollution-monitoring-anomaly-detection/Assets/Data/2015_months_DebitDoseA.csv")

df_train=df.iloc[0:2500]['24/02/2015 11:20']
X_train = df_train.to_numpy()


def create_sequence(seq ,ls ,nseq):

    l=[]

    for i in range (nseq):
        l.append(seq[i:i+ls])

    return np.array(l)


X = create_sequence(X_train,100,2400)
 
# threshold determination   : The first method consist of making the assumption that the errors are gaussian and calculate threshold 


def classify( x_test ,model ,thres, x_val , y_val):

    errors = np.abs(model.predict(x_val)-y_val)

    mu=np.mean(errors)
    sigma= np.std(errors)

    p = [norm.pdf(x, mu , sigma) for x in x_test]

    prediction =np.array([ 1 if p> thres else 0 ])

    return prediction






 





