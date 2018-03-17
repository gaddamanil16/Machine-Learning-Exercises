
# coding: utf-8

# In[1]:


import numpy as np
from numpy.linalg import inv
import csv
import math


# In[2]:


#Datasets
mean0, mean1, cov_given = [],[],[]


#Importing Mean,Covariances from Files

with open("./Datasets/DS1_m_0.txt", 'rb') as csv_mean0_file:
    mean0_data = csv.reader(csv_mean0_file, delimiter = ',')
    for row in mean0_data:
        mean0.append(row)
    del(row)

with open("./Datasets/DS1_m_1.txt", 'rb') as csv_mean1_file:
    mean1_data = csv.reader(csv_mean1_file, delimiter = ',')
    for row in mean1_data:
        mean1.append(row)
    del(row)

with open("./Datasets/DS1_Cov.txt", 'rb') as csv_cov_file:
    cov_data = csv.reader(csv_cov_file, delimiter = ',')
    for row in cov_data:
        cov_given.append(row)
    del(row)
  


# In[3]:


mean1 = np.delete(mean1, len(mean1[0])-1, 1)
mean1_given = (mean1[0]).astype(np.float)
mean0= np.delete(mean0, len(mean0[0])-1, 1)
mean0_given = (mean0[0]).astype(np.float)
cov_given =  (np.delete(cov_given, len(cov_given[0])-1, 1)).astype(np.float)



# In[4]:


#Generating data for class 0
class0 = np.random.multivariate_normal(mean0_given,cov_given, 2000)
#Generating data for class 1
class1 = np.random.multivariate_normal(mean1_given,cov_given, 2000)


# In[5]:


#Calculating Mean and Covariance for class 0 and class 1
mean0_new, mean1_new = [],[]
for i in range(len(class0[0])):
    mean0_new.append(np.mean(class0[:,i]))
    mean1_new.append(np.mean(class1[:,i]))

#Defining mean and Covariance    
mean0 = np.array(mean0_new, dtype = 'float')
mean1 = np.array(mean1_new, dtype = 'float')

param_cov0 =  (1/float(len(class0)))*(np.dot(np.array(class0-np.array(mean0.T)).T,(np.array(class0-np.array(mean0.T)))))
param_cov1 =  (1/float(len(class1)))*(np.dot(np.array(class1-np.array(mean1.T)).T,(np.array(class1-np.array(mean1.T)))))

cov = (float(len(class0))/float(len(class0)*2))*param_cov0 + (float(len(class0))/float(len(class0)*2))* param_cov1


# In[6]:


#Labelling Class 0
class0 = np.hstack((class0, np.zeros((len(class0) ,1))))

#Labelling Class 1
class1 = np.hstack((class1, np.ones((len(class1),1))))


# In[7]:


#Combining data of both classes
raw_data = np.append(class0, class1, axis = 0)

#Shuffling arrays
np.random.shuffle(raw_data)


# In[8]:


#Creating training set
X_train = raw_data[:int(len(raw_data)*0.7)]
Y_train = X_train[:,-1]
X_train = np.delete(X_train, -1, 1)

#Creating test set
X_test = raw_data[int(len(raw_data)*0.7):]
Y_test = X_test[:,-1]
X_test = np.delete(X_test, -1, 1)

dataset_DS1 = np.vstack((X_train,X_test))


# In[9]:


np.savetxt(".\Dataset_DS1.csv", dataset_DS1, delimiter=",",fmt = '%s')


# In[10]:


################### Calculating parameters  ########################
prob_C1 = float(len(class0))/float(len(raw_data))
prob_C2 = float(len(class1))/ float(len(raw_data))

params = np.dot(inv(cov),(mean0 - mean1))
param0 = -0.5* np.dot(np.dot((mean0.T),  inv(cov)), mean0)  + 0.5* np.dot(np.dot((mean1.T),  inv(cov)), mean1) + math.log(prob_C1/prob_C2)


# In[11]:



#Finding probability of data being in class 0 for training set and adding to list
train_prob_c1_x = []

for row in X_train:
    pred = np.dot(params.T, row) + param0
    train_prob_c1_x.append(1/(1 + math.exp(-pred)))

#Lists used to keep count of data in respective classes
train_count_C1, train_count_C2 = [],[]

for value in train_prob_c1_x:
    if(value >= 0.5):
        train_count_C1.append(value)
    else:
        train_count_C2.append(value)


#Finding probability of data being in class 0 for test set and adding to list
test_prob_c1_x = []

for row in X_test:
    pred = np.dot(params.T, row) + param0
    test_prob_c1_x.append(1/(1 + math.exp(-pred)))

#Lists used to keep count of test data in respective classes
test_count_C1, test_count_C2 = [],[]
pred_class = []
test_TP, test_TN, test_FP, test_FN = 0,0,0,0

for value in test_prob_c1_x:
    if(value >= 0.5):
        test_count_C1.append(value)
        pred_class.append(0)
    else:
        test_count_C2.append(value)
        pred_class.append(1)


#Calculating TP,TN, FP, FN
for i in range(len(pred_class)):
    if(pred_class[i] == Y_test[i]):
        if(pred_class[i] == 0):
            test_TP += 1
        if(pred_class[i] == 1):
            test_TN += 1    
    else:
        if(pred_class[i] == 0):
            test_FP += 1
        if(pred_class[i] == 1):
            test_FN += 1
            
accuracy = (float(test_TP + test_TN))/(float(len(pred_class)))*100
precision =  (float(test_TP))/(float(test_TP + test_FP))*100
recall = (float(test_TP))/(float(test_TP + test_FN))*100
f_score = (2*precision*recall)/(precision + recall)

print(accuracy)
print(precision)
print(recall)
print(f_score)

