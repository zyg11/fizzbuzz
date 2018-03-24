from keras.utils import np_utils
from load_data import fizz_buzz
from keras.models import Sequential
from keras.layers.core import Activation,Dense,Dropout
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam
from keras.datasets import mnist
from  keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
import  tensorflow as tf
x_train,y_train=fizzbuzz(101,1000)
x_test,y_test=fizeebuzz(1,100)

model=Sequential()
model.add(Dense(input_dim=10,units=10000,activation='relu'))
model.add(Dense(units=4,Activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=20,nb_epoch=100)

result=model.evaluate(x_test,y_test,batch_size=1000)
print('Acc',format(result[1],'0.2f'))