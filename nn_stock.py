import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#  ~  73cHy9_1XF5Qkui5k_Hp my quandl key ~

df = quandl.get('WIKI/AAPL')
print("connected")

data = df.reset_index()
dfdate = data[[ 'Date']]
print(dfdate)


df = df[['Adj. Open', 'Adj. High', 'Adj. Close', 'Adj. Volume',]]

#print(df['Adj. High'].tail(n=5))
#print(df['Adj. Close'].tail(n=5))

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_CHANGE', 'Adj. Volume']]
forecast_col = 'Adj. Close'


df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.25 * len(df)))

df['label'] = df[forecast_col]#.shift(-forecast_out)
df.dropna(inplace=True)

#print(df.tail(n=5))
days = 150  #HOW MANY DAYS TO TRAIN


df = df[-days:]
print(df)

date = np.array(dfdate[-38:])
#print(dfdate.str.split('T').str[0])# ~~~~~~~~~~~ DATES FIX




#lastdata = int(math.ceil(0.25 * len(df[-days:])))

close = np.array(df['Adj. Close'])
plt.plot(date,close[-38:],'g')

#plt.show()

#exit()

X = np.array(df.drop(['label'], 1))
Y = np.array(df['label'])
X = preprocessing.scale(X)
print(len(X))
print(len(Y))
print('EEEEEEEEEE')

X_scale = MinMaxScaler()
Y_scale = MinMaxScaler()

X = X_scale.fit_transform(X)
Y = Y_scale.fit_transform(Y.reshape(-1,1))



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False)


print(f"\n THIS IS X_TRAIN : \n      {X_train} \n THIS IS Y_TRAIN \n      {Y_train}\n  \n   AND THEIR LENGHTS ARE {len(X_train)} AND {len(Y_train)}\n")
print(f"\n THIS IS X_TEST : \n     {X_test} \n THIS IS Y_TEST \n      {Y_test}\n   \n   AND THEIR LENGHTS ARE {len(X_test)} AND {len(Y_test)}")


X_train = X_train.reshape((-1,1,4))
X_test = X_test.reshape((-1,1,4))

# creating model using Keras
# tf.reset_default_graph()


model_name = 'stock_price_GRU'

model = Sequential()
model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(1, 4)))
model.add(Dropout(0.2))
model.add(GRU(units=256))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam')

model = load_model("{}.h5".format(model_name))
print("MODEL-LOADED")

model.fit(X_train,Y_train,batch_size=250, epochs=500, validation_split=0.1, verbose=1)
model.save("{}.h5".format(model_name))
print('MODEL-SAVED')

score = model.evaluate(X_test, Y_test)
print('Score: {}'.format(score))
yhat = model.predict(X_test)
yhat = Y_scale.inverse_transform(yhat)
Y_test = Y_scale.inverse_transform(Y_test)

print(Y_test)
print(yhat)

plt.plot(date,Y_test,'b')
plt.plot(date,yhat,'r')

#plt.plot(yhat[-100:], label='Predicted')
#plt.plot(Y_test[-100:], label='Ground Truth')
#plt.legend()
plt.show()
