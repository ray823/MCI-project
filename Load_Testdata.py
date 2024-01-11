"""
Created on Thu Oct 19 17:11:29 2023

Author: NTU,BME,OEMAL, JR.LEE, 2023/11/7

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd
from sympy import Symbol
import scipy.io
from mat4py import loadmat
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import time
from keras.models import load_model


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

  
if __name__ == '__main__':
   #Feature_totally = main()
   #主要文件路徑
   base_directory = r"./train_test_split/STtest_1/"
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
         x = daul[i]['ST']
         daul_task_files_list.append(x.T)
         daul_task_files_array = np.array(daul_task_files_list)
         #此處取Y值
         xx = daul[i]['MCI']
         daul_task_files_list_Y.append(xx)
         daul_task_files_array_Y = np.array(daul_task_files_list_Y)
         
       for i in range(len(single)):  
           y = single[i]['ST']
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
   
   
   np.savez('./Array_npz_buffer/TestData1.npz', array1 = final_array, array2 = final_array_YYY)

   
   