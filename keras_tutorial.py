from keras.layers import Input, Dense
from keras.models import Model
import csv
import numpy as np
from keras.optimizers import Adam

train_lables=[]
train_samples=[]

with open('x_data.csv','r') as file:
    reader=csv.reader(file)

    for i in reader:
        train_samples.append(i)

with open('y_data.csv','r') as file:
    reader=csv.reader(file)

    for i in reader:
        train_lables.append(i)


train_samples=np.array(train_samples)
train_lables=np.array(train_lables)
train_samples=np.delete(train_samples,0,axis=0)
train_lables=np.delete(train_lables,0,axis=0)
temp_train_data=[]
temp_label_data=[]
for i in range(0,943):
    for j in range(0,2):
        temp_train_data.append(train_samples[i,j])
train_samples=np.array(temp_train_data)
for i in range(0,943):
    for j in range(0,2):
        temp_label_data.append(train_lables[i,j])
train_lables=np.array(temp_label_data)
train_lables=train_lables.astype('float64')



# this returns a tensor
inputs = Input(shape=(1,))

output_1 = Dense(64,activation='relu')(inputs)
output_2 = Dense(64, activation='relu')(output_1)
prediction = Dense(1, activation='softmax')(output_2)

#this creates a model that includes
#the input layer and three Dense layers
model=Model(inputs=inputs, outputs=prediction)
model.compile(optimizer='rmsprop',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
model.fit(train_samples,train_lables) #starts training
