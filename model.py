
import os
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM ,Dropout , Dense 
import pandas as pd 
import numpy as np 
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

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

df_train=pd.read_csv("Radiological-pollution-monitoring-anomaly-detection/Assets/Data/Normal_signals.csv")
X=df_train['Normal 0'].to_numpy() # contains 3105 samples 

df_test=pd.read_csv("Radiological-pollution-monitoring-anomaly-detection/Assets/Data/2015_months_DebitDoseA.csv")
X_test=df_test['20/10/2015 09:18'].to_numpy()  #contains all data for the last monthn
 
data=np.concatenate([X,X_test],axis=0)
data= data.reshape(-1, 1)

scaler=MinMaxScaler((0,1))
scaled_data= scaler.fit_transform(data)

X,X_test = scaled_data[0:3105],scaled_data[3105:] 

def create_sequence(seq ,ls ):

    X=[]
    y=[]
    for i in range (len(seq) -ls):
        X.append(seq[i:i+ls])
        y.append(seq[i+ls])

    return np.array(X) ,np.array(y)


x,y =create_sequence(X , ls= 100)

x_train ,x_val ,y_train ,y_val = train_test_split( x ,y ,test_size= 0.2 )

x_test,y_test=create_sequence(X_test , ls= 100)


# TRAIN AND SAVE THE MODEL 
"""model=create_model(80 ,80 )
model.summary()

model = train_model(model,x_train,y_train,50)
model.save('LSTM.h5')"""

model=tf.keras.models.load_model('LSTM.h5' ,compile=False
)
model.compile(optimizer='adam', loss='mse')

# threshold determination   : The first method consist of making the assumption that the errors are gaussian and calculate threshold 

class normal_threshold:

    def __init__(self , X_val ,Y_val ,model ,thres =0.9) :

        self.model=model
        self.errors=model.predict(X_val)-Y_val
        self.mu =np.mean(self.errors)
        self.sigma =np.std(self.errors)
        self.eps = norm.pdf(norm.ppf(thres, loc=self.mu, scale=self.sigma) ,  self.mu  ,  self.sigma)
    

    def classify( self , X_test ,Y_test ):

        errors=self.model.predict(X_test) - Y_test

        p = np.array([norm.pdf(x, self.mu , self.sigma)[0]  for x in errors])
        prediction =np.array([ p< self.eps ])

        return prediction , p , self.eps

threshold = normal_threshold(x_val,y_val,model, thres= 0.95)
prediction ,p, eps  =threshold.classify(x_test,y_test)

plt.plot (X_test[100:500])
plt.xlabel('signal time')
plt.ylabel('signal value')
 

#prob=100*[0] + prediction
plt.plot(p[0:400]/10)
plt.axhline( eps /10)
plt.xlabel('signal time')
plt.ylabel('signal proba')
plt.show()


# second thresholding method 



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
    



    




 





