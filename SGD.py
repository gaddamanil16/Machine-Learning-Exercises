
# coding: utf-8

# In[1]:


'''
    COMP 551 Assignment 1 Question 2
    Anil Gadddam
    260776846
'''

#Import packages
import numpy as np
import csv as csv
import matplotlib.pyplot as plt


# In[2]:


#Setting up training data
data = []
result = []
filename = '.\Datasets\Dataset_2_train.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
for row in reader:
        data.append(float(row[0]))
        result.append(float(row[1]))


# In[3]:


#Setting up validation data
valid_data = []
valid_result = []
filename_valid = '.\Datasets\Dataset_2_valid.csv'
valid_raw_data = open(filename_valid, 'rt')
reader_valid = csv.reader(valid_raw_data, delimiter=',', quoting=csv.QUOTE_NONE)

for row in reader_valid:
    valid_data.append(float(row[0]))
    valid_result.append(float(row[1]))  


# In[4]:


#Adding bias column and reshaping for training data
data_ones = np.ones((len(data)), dtype = np.float64).reshape(-1,1)
x_train = np.array(data).reshape(-1,1)
y_train = np.array(result)


# In[5]:


#Adding bias column and reshaping for validation data
x_valid = np.array(valid_data).reshape(-1,1)
y_valid = np.array(valid_result)


# In[6]:


#Intializing the parameters
coef1,coef2 = 0,0


# In[8]:


#Setting up the initial list and leaning rate 
mseTrainingValues,mseValidValues,epochValsTest,mseTestingValues = [],[],[],[]
learning_rate = 1e-6
n_epoch = 50000
epochVals = []

for epoch in range(0,n_epoch):
    MSE_train = 0.0
    epochVals.append(epoch)     

    for i in range(0,len(x_train)):
        #Updating coefficients simultaneously
        coef1 = (coef1 - learning_rate*((coef1 + coef2*x_train[i]) - y_train[i]))
        coef2 = coef2 - learning_rate*((coef1 + coef2*x_train[i]) - y_train[i])* x_train[i]
        
        #Calculating MSE for training data
        Y_train_pred = coef1 + coef2*x_train[i]
        Y_train_actual = y_train[i]
        MSE_train += ((Y_train_pred - Y_train_actual)**2)
        
    MSE_train = MSE_train/len(x_train)
    mseTrainingValues.append(MSE_train)
    MSE_valid = 0.0

    for k in range(len(x_valid)):
        #Calculating MSE for validation data
        Y_valid_pred = coef1 + coef2*x_valid[k]
        Y_valid_actual = y_valid[k]
        MSE_valid += ((Y_valid_pred - Y_valid_actual)**2)
    mseValidValues.append(MSE_valid/len(x_valid))


# In[9]:


#Plotting the nature of the MSE with epochs
line1, = plt.plot(epochVals,mseTrainingValues,'r*')
line2, = plt.plot(epochVals,mseValidValues,'b-')
plt.show()
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('Training and Validation Set Results with Epochs')
plt.legend([line1, line2], ["Training Result", "Validation Result"])




# In[10]:


#Setting up test data
test_data = []
test_result = []
filename_test = '.\Datasets\Dataset_2_test.csv'
test_raw_data = open(filename_test, 'rt')
reader_test = csv.reader(test_raw_data, delimiter=',', quoting=csv.QUOTE_NONE)

for row in reader_test:
    test_data.append(float(row[0]))
    test_result.append(float(row[1]))


# In[20]:


x_test = np.array(test_data).reshape(-1,1)
y_test = np.array(test_result).reshape(-1,1)


# In[21]:


epochValsTest =[]
testedValues =[]
MSE_test_set = 0.0
for row in range(0,len(x_test)):
    Y_predict_test = coef1 + coef2*x_test[i]
    MSE_test_set += (Y_predict_test - y_test[row])**2
    testedValues.append(Y_predict_test)
MSE_test_set/len(x_test)


# In[ ]:


line5, = plt.plot(y_test,'r*')
line6, = plt.plot(testedValues,'b-')
plt.xlabel('Data')
plt.ylabel('Result')
plt.legend()
plt.show([line5, line6], ["Labeled Output", "Test Set Result"])
plt.title('Test Set Results')


# In[18]:


print x_test.shape
print y_test.shape


# In[22]:


coef1 + coef2*x_test[1]

