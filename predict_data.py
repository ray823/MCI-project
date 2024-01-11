import os
import numpy as np
import matplotlib
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
from keras.models import load_model
import time
from tensorflow.keras import backend as K
from IMU_MCI_LSTM_1121_train_test_split import binary_crossentropy_custom


data = np.load('./Array_npz_buffer/TestData1.npz')
test_x = data['array1']
y_test = data['array2']



#DT_array = np.load('DT_array.npy')
X_test_unreshape = (test_x - test_x.mean()) / test_x.std()
X_test = np.transpose(X_test_unreshape, (0, 2, 1))  

'''
daul_mat  = scipy.io.loadmat("./dual_aws.mat")
y_test = daul_mat['Aws'].T
'''

model = load_model('./IMU_LSTM_Model_Performance/STTest1/model.h5')
#new
#model = load_model('./IMU_LSTM_Model_Performance/Test14/model.h5', custom_objects={'binary_crossentropy_custom': binary_crossentropy_custom})


#在測試集上的擬合结果
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
testing_test = time.time()
y_test_predict=model.predict(X_test)
y_test_predict=y_test_predict[:,0]
y_test_predict>0.5
y_test_predict=[int(i) for i in y_test_predict>0.5]
y_test_predict=np.array(y_test_predict)
prediction_model_time = time.time()
#print('Making Predictions took: ', prediction_model_time - fitting_model_time)
#from sklearn import metrics
# Print or use the confusion matrix as needed
print("Confusion Matrix_on_testing_model:")
# Compute the confusion matrix
confusion_test = confusion_matrix(y_test, y_test_predict)

#print(confusion_test)
#print("Classification Report:")
#print(classification_report(y_test, y_test_predict))
fitting_testing_test = time.time()
print('Testing Model_test_data took: ', fitting_testing_test - testing_test)

# 5. Generate a classification report
report = classification_report(y_test, y_test_predict, target_names=['Class 0', 'Class 1'], zero_division=0)
print(report)

# 指定要保存的文件路径
file_path = './IMU_LSTM_Model_Performance/STTest1/classification_report.txt'

# 打开文件并将 classification_report 结果写入文件
with open(file_path, 'w') as file:
    file.write(report)

print("Classification report saved to", file_path)

# 5. 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('./IMU_LSTM_Model_Performance/STTest1/ConfusionMatrix_test.png')
plt.show()




#predictions = model.predict(DT_array)


