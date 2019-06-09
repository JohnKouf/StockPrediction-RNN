# StockPrediction-RNN
Stock Prediction using Recurrent Neural Networks.


In order to run succesfully the user must train the RNN with the following lines being on comment

model = load_model("{}.h5".format(model_name))
print("MODEL-LOADED")

and train with shuffle = True

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False)

Running it 2 or 3 times should be enough for the training (uncommenting the load_model function)

After the training is complete the user must run with shuffle = False  disabling the final training based on the saved data.

#model.fit(X_train,Y_train,batch_size=250, epochs=500, validation_split=0.1, verbose=1)
#model.save("{}.h5".format(model_name))
#print('MODEL-SAVED')
