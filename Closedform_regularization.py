
# coding: utf-8

# In[22]:


'''
    COMP 551 Assignment 1 Question 3
    Anil Gadddam
    260776846
'''

import numpy as np
import csv as csv
import matplotlib.pyplot as plt


# In[23]:


#Setting up training data
data1 = []
data2 =[]
result = []
filename = '.\Datasets\communities.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=",")
for row in reader:
        data1.append(row[0:3])
        data2.append(row[4:])
        #result.append(float(row[-1]))
        
data_ones = np.ones((len(data1)), dtype = np.float64).reshape(-1,1)
x1 = np.array(data1)
x2 = np.array(data2)
x_train = np.hstack((x1,x2))
x_train = np.hstack((data_ones,x_train))



# In[24]:


for i in range((x_train.shape[1])):
    missing_temp = []
    temp_data = x_train[:,i]
    temp = temp_data[(temp_data != '?')]
    temp_2 = temp.astype(np.float)
    avg = np.mean(temp_2)
    
    for counter in range(len(x_train[:,i])):
        if ((x_train[counter,i] == '?')):
            x_train[counter,i] = avg   

np.savetxt(".\Dataset_filled.csv", x_train, delimiter=",",fmt = '%s')    


# In[25]:


def closed_form(x,y):
    xTx = x.T.dot(x)
    XtX = np.linalg.inv(xTx)
    XtX_xT = XtX.dot(x.T)
    theta = XtX_xT.dot(y)
    return theta
    


# In[26]:


def closed_form_regularized(x,y,length,val):
    reg_term=np.identity(length)
    reg_term[0][0] = 0
    xTx = x.T.dot(x) + val*(reg_term)
    XtX = np.linalg.inv(xTx)
    XtX_xT = XtX.dot(x.T)
    theta = XtX_xT.dot(y)
    return theta


# In[27]:


lambda_values = []
rangeOfregfactor = np.arange(0,3,0.01)
avg_MSE_dataset_train_values = 0
avg_MSE_dataset_valid_values = 0
MSE_dataset_valid_values =[]
MSE_dataset_train_values =[]
MSE_avg_regularized =[]

#Clear files
file_name = open('.\coefficients_with_reg.txt','w')
file_name.close
file_name = open('.\coefficients_without_reg.txt','w')
file_name.close

for k in rangeOfregfactor:
    MSE_dataset_test_values_reg =[]
    MSE_dataset_train_reg_values =[]
    #k=0
    for i in range(0,5):
        dataset_testing = np.array(x_train[i*(len(x_train)/5):(i+1)*(len(x_train)/5)],dtype = float) 
        data_set_train = np.array(np.vstack((x_train[:(i)*(len(x_train)/5)],x_train[(i+1)*(len(x_train)/5):])),dtype =float)
        
        #dataset_x1_train = np.array(dataset_folded[:int(0.8*len(dataset_folded))],dtype = float) 
        dataset_y1_train = np.array(data_set_train[:,-1]) 
        data_set_train = np.delete(data_set_train,len(data_set_train[1])-1,1)

        #dataset_x1_valid = np.array(dataset_folded[int(0.8*len(dataset_folded)):],dtype = float)
        dataset_y1_testing = np.array(dataset_testing[:,-1]) 
        dataset_testing = np.delete(dataset_testing,len(dataset_testing[1])-1,1)
        
        #Training without regularization
        if k==0:
            theta = closed_form(data_set_train,dataset_y1_train)
            
            file_name = open('.\coefficients_without_reg.txt','a')
        
            for j in theta:
                file_name.write("%.18f\t" % j)
            file_name.write('\n****************\n\n')
            file_name.close
            
            MSE_dataset_valid = 0
            MSE_dataset_train =0
            
            MSE_dataset_train = np.mean((dataset_y1_train- np.dot(data_set_train,theta))**2)
            
            MSE_dataset_valid = np.mean((dataset_y1_testing- np.dot(dataset_testing,theta))**2)
            
            #MSE_dataset_train_values.append(MSE_dataset_train)
            MSE_dataset_valid_values.append(MSE_dataset_valid)
            
        file_name = open('.\coefficients_without_reg.txt','a')
        file_name.write("MSE : %f\n\n****************\n" % np.mean(MSE_dataset_valid_values))
        file_name.close
        # Ridge regression
        theta_regularized = closed_form_regularized(data_set_train,dataset_y1_train,len(data_set_train[1]),k)
        
        MSE_dataset_train_reg = 0
        MSE_dataset_test_reg = 0
        
        training_prediction_with_reg = np.dot(data_set_train,theta_regularized)
        test_prediction_with_reg = np.dot(dataset_testing,theta_regularized)    
        
        for j in range(0,len(training_prediction_with_reg)):
            MSE_dataset_train_reg += ((dataset_y1_train[j] - training_prediction_with_reg[j])**2)
            
        MSE_dataset_train_reg /= len(training_prediction_with_reg)
                
        #Test error for ridge regression
        for j in range(0,len(test_prediction_with_reg)):
            MSE_dataset_test_reg += ((dataset_y1_testing[j] - test_prediction_with_reg[j])**2)
            
        MSE_dataset_test_reg /= len(test_prediction_with_reg)
        MSE_dataset_test_values_reg.append(MSE_dataset_test_reg)
        
        file_name = open('.\coefficients_with_reg.txt','a')
        for j in theta_regularized:
            file_name.write("%.18f\t" % j)
        file_name.write('\n****************\n')
        file_name.close
    
    print('Test error with regularization: %f lambda %f' %(np.mean(MSE_dataset_test_values_reg),k))
    MSE_avg_regularized.append(np.mean(MSE_dataset_test_values_reg))    
    lambda_values.append(k)
    
    file_name = open('.\coefficients_with_reg.txt','a')
    file_name.write("MSE : %f\n\n****************\n" % np.mean(MSE_dataset_test_values_reg))
    file_name.close

# Mean Test error without regularization
print("MSE without regularization  %f" %(np.mean(MSE_dataset_valid_values)))

#Getting lambda 
print('Lambda for best fit : %f ,MSE: %f' %(lambda_values[MSE_avg_regularized.index(min(MSE_avg_regularized))],min(MSE_avg_regularized)))

