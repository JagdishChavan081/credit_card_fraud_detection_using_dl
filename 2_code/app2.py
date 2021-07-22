import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split

st.title("Credit Card Fraud Detection!")
st.image("/home/jagdish/pytn/7_projects/credit/card.png")

df=pd.read_csv('/home/jagdish/pytn/7_projects/credit/3_dset/dataset.csv')

# balance the dataset
fraud  = df[df["Class"]==1]
non_fraud = df[df["Class"]==0]

#matrix of features
x = df.drop(labels=['Class'], axis=1)
#dependent variable
y = df['Class']

#spliting the data set in train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

#reshaping the dataset
x_train=x_train.reshape(787, 30, 1 )
x_test=x_test.reshape(197, 30, 1)

#define object
model = tf.keras.models.Sequential()

#first CNN layer
#activation functions bring non linearity to model 
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, padding='same',activation='relu',input_shape=(30,1)))

#batch normalization 
model.add(tf.keras.layers.BatchNormalization())

#maxpool layer
#pooling layer reduces the dimensionality of op
model.add(tf.keras.layers.MaxPool1D(pool_size=2))

#droupout layer
#regularization technique
#ignoring 20% neuron while training randomly
model.add(tf.keras.layers.Dropout(0.2))

# Second CNN layer
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding='same',activation='relu',input_shape=(30,1)))

#batch normalization 
model.add(tf.keras.layers.BatchNormalization())

#maxpool layer
model.add(tf.keras.layers.MaxPool1D(pool_size=2))

#droupout layer
model.add(tf.keras.layers.Dropout(0.3))

#flatten layer
#converts array to vector
model.add(tf.keras.layers.Flatten())

#first dense layer

model.add(tf.keras.layers.Dense(units=64, activation="relu"))
#droupout layer
model.add(tf.keras.layers.Dropout(0.3))

#outputlayer
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=25,
                    validation_data=(x_test, y_test))

y_pred = model.predict_classes(x_test)


with st.form(key='my_form'):
    num=st.number_input(label='Enter the CVV')
    submit_button=st.form_submit_button(label='Predict')
    if submit_button:
        num=int(num)
        y=y_pred[num]
        if y==0:
            st.write("Transaction is Non Fradulant")
        if y==1:
            st.write("Transaction is Fradulant")
