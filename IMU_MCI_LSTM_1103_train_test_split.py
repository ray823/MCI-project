"""
Created on Thu Oct 19 17:11:29 2023

Author: NTU,BME,OEMAL, JR.LEE, 2023/10/19

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import random
import openpyxl
import pandas as pd
from sympy.plotting import plot3d
from IPython.display import display
from sympy import Symbol
import scipy.io
from mat4py import loadmat
import keras
from keras.callbacks import History,EarlyStopping
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
import time
from keras.models import load_model
from scipy.interpolate import interp1d
''' 
start = time.time()

'''
def Norm (mat_path, Frames):
    Input  = scipy.io.loadmat(mat_path)
    Output = []
    Input = np.array(Input['IA_Results'])
    xdata = np.linspace(0,Input.shape[0]-1,Frames)
    for i in range (Input.shape[1]):
        x_frame = np.linspace(0,Input.shape[0]-1,Input.shape[0])
        y_strain = Input[:,i]
        y_f = interp1d(x_frame,y_strain,'cubic')
        ydata = y_f(xdata)
        Output.append(ydata)
        
    return np.array(Output).T

def process_directory(directory):
    daul_task_files = []
    daul_file_path = []
    single_task_files = []
    single_file_path = []
    for root, dirs, files in os.walk(directory):
        if files is not None:
            for file in files:
                if file.endswith(".mat") and "IA" not in file:
                    file_path = os.path.join(root, file)
                    
                    if "daul_task" in root:
                        mat_data = scipy.io.loadmat(file_path)
                        #print(file_path)
                        daul_file_path.append(file_path)
                        daul_task_files.append(mat_data)
                        
                        
                        
                    elif "single_task" in root:
                        mat_data1 = scipy.io.loadmat(file_path)
                        single_file_path.append(file_path)
                        single_task_files.append(mat_data1)
                       
          
    return daul_task_files,single_task_files,daul_file_path,single_file_path

def process_directory_for_this(directory):
    daul_task_files = []
    daul_file_path = []
    single_task_files = []
    single_file_path = []
    for root, dirs, files in os.walk(directory):
        if files is not None:
            for file in files:
                if file.endswith(".mat") and "IA" not in file:
                    file_path = os.path.join(root, file)
                    
                    if "daul_task" in root:
                        mat_data = scipy.io.loadmat(file_path)
                        daul_task_files.append(mat_data)
                        
                        
                        
                    elif "single_task" in root:
                        mat_data1 = scipy.io.loadmat(file_path)
                        single_task_files.append(mat_data1)
                       
          
    return daul_task_files,single_task_files

def Model_Training(final_array,final_array_Y):
    '''
    import tensorflow as tf
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    '''
    
    
    #We split data as 8:1:1
    X_normalized = (final_array - final_array.mean()) / final_array.std()
    #X_train, X_temp, y_train, y_temp = train_test_split(X_normalized, final_array_Y, test_size=0.2, random_state=42)
    #X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_normalized, final_array_Y, test_size=0.2, random_state=42)
    
    
    
   
    
    # reshape input to be [samples, time steps, features] which is required for LSTM
    #X_train =X_train.reshape(X_train.shape[0],X_train.shape[2] , X_train.shape[1])
    #X_val =X_val.reshape(X_val.shape[0],X_val.shape[2] , X_val.shape[1])
    #X_test = X_test.reshape(X_test.shape[0],X_test.shape[2] , X_test.shape[1])
    X_train = np.transpose(X_train, (0, 2, 1))  
    X_val = np.transpose(X_val, (0, 2, 1))  
    # Record the start time
    start_fitting = time.time()
    
    
    d = 0.01
    model = Sequential()
    #model.add(Bidirectional(GRU(64, input_shape=(500,4))))
    model.add(Bidirectional(GRU(64, input_shape=(101,12))))
    model.add(Dropout(d))
    #model.add(SelfAttention(64))
    model.add(Dense(32,activation='relu'))        
    model.add(Dense(1,activation='sigmoid'))
    
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer= opt,metrics=['accuracy'])
    start_fitting = time.time()
    #earlystopping
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, verbose=2, mode='auto', restore_best_weights=True)
    model.fit(X_train, y_train, epochs = 100, batch_size = 32,validation_data=(X_val, y_val),callbacks=[monitor]) 
    
    '''
    d = 0.01
    model = Sequential()
    #input_shape=X_train.shape
    model.add(LSTM(64, input_shape=(101,12)))
    model.add(Dropout(d))
    #model.add(SelfAttention(64))
    model.add(Dense(32,init='uniform',activation='relu'))        
    model.add(Dense(1,init='uniform',activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer= opt,metrics=['accuracy'])
    #earlystopping
    start_fitting = time.time()
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, verbose=2, mode='auto', restore_best_weights=True)
    model.fit(X_train, y_train, nb_epoch = 2000, batch_size = 32,validation_data=(X_val, y_val),callbacks=[monitor]) #训练模型1000次
    '''
    
    epochxxx =  monitor.stopped_epoch+1
    fitting_model_time = time.time()
    print('Training Model took: ', fitting_model_time - start_fitting)
    
    input_shape=(101,12)
    model.build(input_shape)
    print(model.summary())
    with open('./IMU_LSTM_Model_Performance/Test4/modelsummary.txt', 'w') as f:
    
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    #保存模型和權重
    saved = model.save('./IMU_LSTM_Model_Performance/Test4/model.h5')    
    model.save_weights('./IMU_LSTM_Model_Performance/Test4/model_weights.h5')  # to store


    #画出迭代loss和acc曲线
    pd.DataFrame(model.history.history).plot()
    plt.title('Loss and Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    # 保存图形到文件（以PNG格式保存）
    plt.savefig('./IMU_LSTM_Model_Performance/Test4/loss_acc_curves.png')





    
    loss_train = model.history.history['loss']
    loss_val = model.history.history['val_loss']
    epochs = range(epochxxx)
    fig = plt.figure()
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    fig.savefig('./IMU_LSTM_Model_Performance/Test4/train_Validation_loss.png')
    
    
    loss_train = model.history.history['accuracy']
    loss_val = model.history.history['val_accuracy']
    epochs = range(epochxxx)
    fig1 = plt.figure()
    plt.plot(epochs, loss_train, 'g', label='Training accuracy')
    plt.plot(epochs, loss_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    fig1.savefig('./IMU_LSTM_Model_Performance/Test4/train_Validation_accuracy.png')
    

    #在訓練集上的擬合结果
    testing_train = time.time()
    y_train_predict=model.predict(X_train)
    y_train_predict=y_train_predict[:,0]
    y_train_predict>0.5
    y_train_predict=[int(i) for i in y_train_predict>0.5]
    y_train_predict=np.array(y_train_predict)
    #from sklearn import metrics
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Print or use the confusion matrix as needed
    print("Confusion Matrix_on_training_model:")
    # Compute the confusion matrix
    confusion_train = confusion_matrix(y_train, y_train_predict)
    print(confusion_train)
    
    print("Classification Report:")
    print(classification_report(y_train, y_train_predict))
    fitting_testing_train = time.time()
    print('Testing Model_train_data took: ', fitting_testing_train - testing_train)
    
    #print("精確度等指標：")
    #print(metrics.classification_report(y_train,y_train_predict))
    #print("混淆矩陣：") 
    #print(metrics.confusion_matrix(y_train,y_train_predict))
    
    '''
    #在測試集上的擬合结果
    testing_test = time.time()
    y_test_predict=model.predict(X_test)
    y_test_predict=y_test_predict[:,0]
    y_test_predict>0.5
    y_test_predict=[int(i) for i in y_test_predict>0.5]
    y_test_predict=np.array(y_test_predict)
    prediction_model_time = time.time()
    print('Making Predictions took: ', prediction_model_time - fitting_model_time)
    #from sklearn import metrics
    # Print or use the confusion matrix as needed
    print("Confusion Matrix_on_testing_model:")
    # Compute the confusion matrix
    confusion_test = confusion_matrix(y_test, y_test_predict)
    print(confusion_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_test_predict))
    fitting_testing_test = time.time()
    print('Testing Model_train_data took: ', fitting_testing_test - testing_test)
    
    #print("精確度等指標：")
    #print(metrics.classification_report(y_test,y_test_predict))
    #print("混淆矩陣：")
    #print(metrics.confusion_matrix(y_test,y_test_predict))
    #end = time.time()
    #print('TOTAL time spent', end-start)
    '''
    
    return X_train,X_val
  
if __name__ == '__main__':
   #Feature_totally = main()
   #主要文件路徑
   base_directory = r"./train_test_split/train/"
   #base_directory = r"./Norm1_Mix/"
   #子文件路徑
   subdirectories = []
   subdirectories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
   print(subdirectories)
  
   # Now you have two lists: dual_task_files and single_task_files
   daul =[]
   single = []
   final_list = []
   #這裡是最終Y的list
   final_list_Y = []
   #計算檔案匯入到值全部取出時間
   t1 = time.time()
   for c in range(len(subdirectories)):
       daul_task_files_list =[]
       single_task_files_list = []
       #-------------------
       #此處為Y的LIST
       daul_task_files_list_Y =[]
       single_task_files_list_Y = []
       
       
       #-------------------
       daul,single = process_directory_for_this(subdirectories[c])  
       for i in range(len(daul)):
         x = daul[i]['SC']
         daul_task_files_list.append(x.T)
         daul_task_files_array = np.array(daul_task_files_list)
         #此處取Y值
         xx = daul[i]['MCI']
         daul_task_files_list_Y.append(xx)
         daul_task_files_array_Y = np.array(daul_task_files_list_Y)
         
       for i in range(len(single)):  
           y = single[i]['SC']
           single_task_files_list.append(y.T)
           single_task_files_array = np.array(single_task_files_list)
           #此處取Y值
           yy = single[i]['MCI']
           single_task_files_list_Y.append(yy)
           single_task_files_array_Y = np.array(single_task_files_list_Y)
       for i in range(daul_task_files_array.shape[0]):
           for j in range(single_task_files_array.shape[0]):
               z = np.concatenate((daul_task_files_array[i],single_task_files_array[j]),axis = 0)
               final_list.append(z)
               final_array = np.array(final_list)
       #zz = daul_task_files_list_Y[0][0]
       #print(zz)
       
       for i in range(daul_task_files_array_Y.shape[0]):
           for j in range(single_task_files_array_Y.shape[0]):
               #print(single_task_files_array.shape[0])
               #此處取Y值
               zz = daul_task_files_list_Y[i][0]
               final_list_Y.append(zz)
               final_array_Y = np.array(final_list_Y)
   count0 = 0
   count1 = 0
   #final_array_YY = int(final_array_Y)
   final_array_YY = final_array_Y.astype(float)
   final_array_YYY = final_array_YY.astype(int)
   for i in range(len(final_array_YYY)):
       if final_array_YYY[i] == 0:
           count0+=1
       elif final_array_YYY[i] ==1:
           count1+=1
   print(count0,count1)
           
           
       
          
   t2 = time.time()
   print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')
   print("Finish")
   X_train,X_val =  Model_Training(final_array,final_array_YYY)         
   
    
        