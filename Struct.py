import os.path
import random
import sys
import time
import pickle

import numpy as np
import pandas as pd

import time
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
#The above packages tensorflow.keras cannot be replaced by keras

#data location
root = "./"
data_folder = root+"dataset/20220111/"
AESHD_trained_models_folder = root+"trained_models/"
history_folder = root+"training_history/"
predictions_folder = root+"model_predictions/"
class_num=17         #HW species of encrypted intermediate value
def mlp_architecture(input_size=140, learning_rate=0.00001, classes=class_num):

    model=tf.keras.Sequential([
        tf.keras.layers.Dense(500, activation='relu', kernel_initializer='he_normal', input_shape=(input_size,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(500, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(500, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(classes, activation='softmax')
        ])
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


def shuffle_data(profiling_x,label_y):
    l = list(zip(profiling_x,label_y))
    random.shuffle(l)
    shuffled_x,shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)

#### Training
def train_model(X_profiling, Y_profiling, X_test, Y_test, model, save_file_name, epochs=150, batch_size=100, max_lr=1e-3):

    # Save model every epoch
    save_model = ModelCheckpoint(save_file_name)

    callbacks = [save_model]
    history = model.fit(x=X_profiling, y=to_categorical(Y_profiling, num_classes=class_num),validation_data=(X_test, to_categorical(Y_test, num_classes=class_num)),batch_size=batch_size, verbose=1, epochs=epochs, callbacks=callbacks)
    return history


nb_epochs = 500
batch_size = 50
learning_rate = 1e-4


start = time.time()

#load data set

(train_trace, train_label)= (np.load(data_folder + 'Train_Wave_20220111_Every_class_all.npy'), np.load(data_folder + 'label_train_20220110_Every_class_all.npy'))
(train_trace, train_label) = shuffle_data(train_trace, train_label)
(Test_trace_aw1,Test_label_aw1) = (np.load(data_folder + 'Test_Wave_20220111_12.npy'), np.load(data_folder + 'label_test_0107_aw1.npy'))
(Test_trace_aw2,Test_label_aw2) = (np.load(data_folder + 'Test_Wave_20220111_23.npy'), np.load(data_folder + 'label_test_0107_aw2.npy'))
(Test_trace_aw3,Test_label_aw3) = (np.load(data_folder + 'Test_Wave_20220111_34.npy'), np.load(data_folder + 'label_test_0107_aw3.npy'))
(Test_trace_aw4,Test_label_aw4) = (np.load(data_folder + 'Test_Wave_20220111_45.npy'), np.load(data_folder + 'label_test_0107_aw4.npy'))
input_size=train_trace.shape[1]

scaler = preprocessing.StandardScaler()
train_trace = scaler.fit_transform(train_trace)
Test_trace_aw1 = scaler.transform(Test_trace_aw1)
Test_trace_aw2 = scaler.transform(Test_trace_aw2)
Test_trace_aw3 = scaler.transform(Test_trace_aw3)
Test_trace_aw4 = scaler.transform(Test_trace_aw4)


scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
train_trace = scaler.fit_transform(train_trace)
Test_trace_aw1 = scaler.transform(Test_trace_aw1)
Test_trace_aw2 = scaler.transform(Test_trace_aw2)
Test_trace_aw3 = scaler.transform(Test_trace_aw3)
Test_trace_aw4 = scaler.transform(Test_trace_aw4)

train_label=train_label.reshape((train_trace.shape[0],1))



# Choose your model
model = mlp_architecture(input_size=input_size, learning_rate=learning_rate)
model_name="MLP"

print("\n############### Starting Training #################\n")

# Record the metrics
history = train_model(train_trace[:int(train_trace.shape[0]*0.9), 0:input_size], train_label[:int(train_trace.shape[0]*0.9)], train_trace[int(train_trace.shape[0]*0.9):, 0:input_size], train_label[int(train_trace.shape[0]*0.9):], model, AESHD_trained_models_folder + model_name, epochs=nb_epochs, batch_size=batch_size)
end=time.time()

print("\n############### Training Done #################\n")

# Save the metrics
with open(history_folder + 'history_' + model_name, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

#################################################
#################################################

####               Prediction              ######

#################################################
#################################################



print("\n############### Starting Predictions #################\n")
prediction = model(Test_trace_aw1)
np.savetxt("result/pro_cho_12.txt",prediction,fmt="%.5f")

prediction = model(Test_trace_aw2)
np.savetxt("result/pro_cho_23.txt",prediction,fmt="%.5f")

prediction = model(Test_trace_aw3)
np.savetxt("result/pro_cho_34.txt",prediction,fmt="%.5f")

prediction = model(Test_trace_aw4)
np.savetxt("result/pro_cho_45.txt",prediction,fmt="%.5f")

print("\n############### Training Done #################\n")