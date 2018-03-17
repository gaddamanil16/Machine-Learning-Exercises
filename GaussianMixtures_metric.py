
# coding: utf-8

# In[25]:


import numpy as np
import csv
from numpy.linalg import inv


# In[26]:


#Importing Mean,Covariances from Files
# Gaussian1
mean0_mixture1,mean1_mixture1,cov_given1_mixture1=[],[],[]

with open("./Datasets/DS2_c1_m1.txt", 'rb') as csv_mean0_file:
    mean0_data = csv.reader(csv_mean0_file, delimiter = ',')
    for row in mean0_data:
        mean0_mixture1.append(row)
    del(row)

with open("./Datasets/DS2_c2_m1.txt", 'rb') as csv_mean1_file:
    mean1_data = csv.reader(csv_mean1_file, delimiter = ',')
    for row in mean1_data:
        mean1_mixture1.append(row)
    del(row)

with open("./Datasets/DS2_Cov1.txt", 'rb') as csv_cov_file:
    cov_data = csv.reader(csv_cov_file, delimiter = ',')
    for row in cov_data:
        cov_given1_mixture1.append(row)
    del(row)


# In[27]:



mean1_mixture1 = np.delete(mean1_mixture1, len(mean1_mixture1[0])-1, 1)
mean1_given_mixture1 = (mean1_mixture1[0]).astype(np.float)
mean0_mixture1= np.delete(mean0_mixture1, len(mean0_mixture1[0])-1, 1)
mean0_given_mixture1 = (mean0_mixture1[0]).astype(np.float)
cov_given_mixture1 =  (np.delete(cov_given1_mixture1, len(cov_given1_mixture1[0])-1, 1)).astype(np.float)


#Generating data for class 0
class0_mixture1 = np.random.multivariate_normal(mean0_given_mixture1,cov_given_mixture1, 2000)
#Generating data for class 1
class1_mixture1 = np.random.multivariate_normal(mean1_given_mixture1,cov_given_mixture1, 2000)

#Labelling Class 0
class0_mixture1 = np.hstack((class0_mixture1, np.zeros((len(class0_mixture1) ,1))))

#Labelling Class 1
class1_mixture1 = np.hstack((class1_mixture1, np.ones((len(class1_mixture1),1))))

rawdata_mixture1 = np.append(class0_mixture1, class1_mixture1, axis = 0)

from_rawdata_mixture1 = rawdata_mixture1[np.random.randint(0,rawdata_mixture1.shape[0],int(0.1*len(rawdata_mixture1)))]


# In[28]:



#Gaussian2
mean0_mixture2,mean1_mixture2,cov_given1_mixture2=[],[],[]
with open("./Datasets/DS2_c1_m2.txt", 'rb') as csv_mean0_file:
    mean0_data_mixture2 = csv.reader(csv_mean0_file, delimiter = ',')
    for row in mean0_data_mixture2:
        mean0_mixture2.append(row)
    del(row)

with open("./Datasets/DS2_c2_m2.txt", 'rb') as csv_mean1_file:
    mean1_data_mixture2 = csv.reader(csv_mean1_file, delimiter = ',')
    for row in mean1_data_mixture2:
        mean1_mixture2.append(row)
    del(row)

with open("./Datasets/DS2_Cov2.txt", 'rb') as csv_cov_file:
    cov_data_mixture2 = csv.reader(csv_cov_file, delimiter = ',')
    for row in cov_data_mixture2:
        cov_given1_mixture2.append(row)
    del(row)

mean1_mixture2 = np.delete(mean1_mixture2, len(mean1_mixture2[0])-1, 1)
mean1_given_mixture2 = (mean1_mixture2[0]).astype(np.float)
mean0_mixture2= np.delete(mean0_mixture2, len(mean0_mixture2[0])-1, 1)
mean0_given_mixture2 = (mean0_mixture2[0]).astype(np.float)
cov_given_mixture2 =  (np.delete(cov_given1_mixture2, len(cov_given1_mixture2[0])-1, 1)).astype(np.float)


#Generating data for class 0
class0_mixture2 = np.random.multivariate_normal(mean0_given_mixture2,cov_given_mixture2, 2000)
#Generating data for class 1
class1_mixture2 = np.random.multivariate_normal(mean1_given_mixture2,cov_given_mixture2, 2000)

#Labelling Class 0
class0_mixture2 = np.hstack((class0_mixture2, np.zeros((len(class0_mixture2) ,1))))

#Labelling Class 1
class1_mixture2 = np.hstack((class1_mixture2, np.ones((len(class1_mixture2),1))))

rawdata_mixture2 = np.append(class0_mixture2, class1_mixture2, axis = 0)

from_rawdata_mixture2 = rawdata_mixture2[np.random.randint(0,rawdata_mixture2.shape[0],int(0.42*len(rawdata_mixture2)))]


# In[29]:



#Gaussian3

mean0_mixture3,mean1_mixture3,cov_given1_mixture3=[],[],[]
with open("./Datasets/DS2_c1_m3.txt", 'rb') as csv_mean0_file:
    mean0_data_mixture3 = csv.reader(csv_mean0_file, delimiter = ',')
    for row in mean0_data_mixture3:
        mean0_mixture3.append(row)
    del(row)

with open("./Datasets/DS2_c2_m3.txt", 'rb') as csv_mean1_file:
    mean1_data_mixture3 = csv.reader(csv_mean1_file, delimiter = ',')
    for row in mean1_data_mixture3:
        mean1_mixture3.append(row)
    del(row)

with open("./Datasets/DS2_Cov3.txt", 'rb') as csv_cov_file:
    cov_data_mixture3 = csv.reader(csv_cov_file, delimiter = ',')
    for row in cov_data_mixture3:
        cov_given1_mixture3.append(row)
    del(row)

mean1_mixture3 = np.delete(mean1_mixture3, len(mean1_mixture3[0])-1, 1)
mean1_given_mixture3 = (mean1_mixture3[0]).astype(np.float)
mean0_mixture3= np.delete(mean0_mixture3, len(mean0_mixture3[0])-1, 1)
mean0_given_mixture3 = (mean0_mixture3[0]).astype(np.float)
cov_given_mixture3 =  (np.delete(cov_given1_mixture3, len(cov_given1_mixture3[0])-1, 1)).astype(np.float)


#Generating data for class 0
class0_mixture3 = np.random.multivariate_normal(mean0_given_mixture3,cov_given_mixture3, 2000)
#Generating data for class 1
class1_mixture3 = np.random.multivariate_normal(mean1_given_mixture3,cov_given_mixture3, 2000)

#Labelling Class 0
class0_mixture3 = np.hstack((class0_mixture3, np.zeros((len(class0_mixture3) ,1))))

#Labelling Class 1
class1_mixture3 = np.hstack((class1_mixture3, np.ones((len(class1_mixture3),1))))


# In[30]:



rawdata_mixture3 = np.append(class0_mixture3, class1_mixture3, axis = 0)

from_rawdata_mixture3 = rawdata_mixture3[np.random.randint(0,rawdata_mixture3.shape[0],int(0.48*len(rawdata_mixture3)))]

datasetDs2 = np.concatenate((from_rawdata_mixture1, from_rawdata_mixture2, from_rawdata_mixture3),axis=0)

np.savetxt(".\Dataset_DS2.csv", datasetDs2, delimiter=",",fmt = '%s')

indices_class0 = np.where(datasetDs2[:,-1] == 0)
indices_class1 = np.where(datasetDs2[:,-1] == 1)
datasetDs2_class0 = np.concatenate(datasetDs2[indices_class0,:],axis=0)
datasetDs2_class1 = np.concatenate(datasetDs2[indices_class1,:],axis=0)



# In[31]:


#Calculating parameters
mean0_new, mean1_new = [],[]
for i in range(len(datasetDs2_class0[0])-1):
    mean0_new.append(np.mean(datasetDs2_class0[:,i]))
for i in range(len(datasetDs2_class1[0])-1):
    mean1_new.append(np.mean(datasetDs2_class1[:,i]))

#Defining mean and Covariance
mean0 = np.array(mean0_new, dtype = 'float')
mean1 = np.array(mean1_new, dtype = 'float')

param_cov0 =  (1/float(len(datasetDs2_class0)))*(np.dot(np.array(datasetDs2_class0[:,:-1]-np.array(mean0.T)).T,(np.array(datasetDs2_class0[:,:-1]-np.array(mean0.T)))))
param_cov1 =  (1/float(len(datasetDs2_class1)))*(np.dot(np.array(datasetDs2_class1[:,:-1]-np.array(mean1.T)).T,(np.array(datasetDs2_class1[:,:-1]-np.array(mean1.T)))))

cov = (float(len(datasetDs2_class0))/float(len(datasetDs2_class0)*2))*param_cov0 + (float(len(datasetDs2_class1))/float(len(datasetDs2_class1)*2))* param_cov1


# In[32]:



#Creating training set
X_train = datasetDs2[:int(len(datasetDs2)*0.7)]
Y_train = X_train[:,-1]
X_train = np.delete(X_train, -1, 1)

#Creating test set
X_test = datasetDs2[int(len(datasetDs2)*0.7):]
Y_test = X_test[:,-1]
X_test = np.delete(X_test, -1, 1)

dataset_DS1 = np.vstack((X_train,X_test))
#np.savetxt(".\Dataset_DS1.csv", dataset_DS1, delimiter=",",fmt = '%s')
#np.savetxt(".\Dataset_DS1.csv", X_test, delimiter=",",fmt = '%s')
################### Calculating parameters  ########################
prob_C1 =len(datasetDs2_class0)/float(len(datasetDs2))
prob_C2 = len(datasetDs2_class1)/ float(len(datasetDs2))

params = np.dot(inv(cov),(mean0 - mean1))
param0 = -(0.5* np.dot(np.dot((mean0.T),  inv(cov)), mean0))  + (0.5* np.dot(np.dot((mean1.T),  inv(cov)), mean1)) + np.log(prob_C1/prob_C2)

#Finding probability of data being in class 0 for training set and adding to list
train_prob_c1_x = []

for row in X_train:
    pred = np.dot(params.T, row) + param0
    train_prob_c1_x.append(1/(1 + np.exp(-pred)))

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
    test_prob_c1_x.append(1/(1 + np.exp(-pred)))

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

print ('Accuracy ' + str(accuracy))
print('Precision ' +str(precision))
print('Recall ' + str(recall))
print('Fscore '+ str(f_score))



# In[10]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline


y = np.array([54.26,64.57,67.27,70,72])
x = np.array([1,3,5,10,15])
xnew = np.linspace(x.min(),x.max(),10)
plt.plot(x,y)
#power_smooth = spline(x,y,xnew)

#plt.plot(xnew,power_smooth)
plt.show()

plt.xlabel('K')
plt.ylabel('F-Score')

