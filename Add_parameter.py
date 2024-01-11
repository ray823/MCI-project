"""
Created on Thu Oct 19 17:11:29 2023

Author: NTU,BME,OEMAL, JR.LEE, 2023/10/19

"""

from IMU_MCI_LSTM_1019 import*
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat



if __name__ == '__main__':  
    base_directory_0 = r"./Norm1_add_parameters/0/"
    base_directory_1 = r"./Norm1_add_parameters/1/"
    subdirectories_0 = []
    subdirectories_1 = []
    subdirectories_0 = [os.path.join(base_directory_0, d) for d in os.listdir(base_directory_0) if os.path.isdir(os.path.join(base_directory_0, d))]
    print(subdirectories_0)
    subdirectories_1 = [os.path.join(base_directory_1, d) for d in os.listdir(base_directory_1) if os.path.isdir(os.path.join(base_directory_1, d))]
    print(subdirectories_1)
    
    # Now you have two lists: dual_task_files and single_task_files
    daul_file_0 =[]
    single_file_0 = []
    for c in range(len(subdirectories_0)):
        daul_0 =[]
        single_0 = []
        #final_daul_0_list = []
        daul_0,single_0,daul_file_0,single_file_0 = process_directory(subdirectories_0[c])  
        for i in range(len(daul_0)):
            file_name = daul_file_0[i]
            new_key = 'MCI'
            new_value = '0'
            daul_0[i][new_key] = new_value
            savemat(file_name,daul_0[i], do_compression=False)
            
            #daul_0[i].setdefault('MCI','0')
            #savemat(file_name,daul_0[i])
            
        
        for i in range(len(single_0)):
            file_name = single_file_0[i]
            new_key = 'MCI'
            new_value = '0'
            single_0[i][new_key] = new_value
            savemat(file_name,single_0[i], do_compression=False)
            
            #single_0[i].setdefault('MCI','0')
            #savemat(file_name,single_0[i])
            
    
    daul_file_1 =[]
    single_file_1 = []
    for c in range(len(subdirectories_1)):
        daul_1 =[]
        single_1 = []
        daul_1,single_1,daul_file_1,single_file_1 = process_directory(subdirectories_1[c])  
        for i in range(len(daul_1)):
            file_name = daul_file_1[i]
            new_key = 'MCI'
            new_value = '1'
            daul_1[i][new_key] = new_value
            savemat(file_name,daul_1[i], do_compression=False)
            '''
            daul_1[i].setdefault('MCI','1')
            savemat(file_name,daul_1[i])
            '''
        for i in range(len(single_1)):
            file_name = single_file_1[i]
            new_key = 'MCI'
            new_value = '1'
            single_1[i][new_key] = new_value
            savemat(file_name,single_1[i], do_compression=False)
            '''
            single_1[i].setdefault('MCI','1')
            savemat(file_name,single_1[i])
            '''
    print("Finish") 