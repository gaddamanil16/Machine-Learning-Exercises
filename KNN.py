
# coding: utf-8

# In[13]:


#import packages
import csv
import numpy as np
import math
import operator


# In[14]:


#importing dataset
dataset = []
x_train = []

with open('.\Dataset_DS1_KNN.csv', 'rb') as csvfile:
    lines = csv.reader(csvfile)
    for row in lines:
        dataset.append(row)
    dataset = np.array(dataset)


# In[15]:


# Creating training and testing sets
np.random.shuffle(dataset)
x_train = np.array(dataset[0:int(len(dataset)*0.7)],dtype = float)
x_test = np.array(dataset[int(len(dataset)*0.7):],dtype=float)


# In[16]:


#Calculating eucledian distance
def euclideanDistance(instance1, instance2, length):
        distance = 0
        for x in range(length):
                distance += np.power((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)


# In[17]:


#Getting neighbors
def getNeighbors(trainingSet, testInstance, k):
        distances = []
        length = len(testInstance)-1
        for x in range(len(trainingSet)):
                dist = euclideanDistance(testInstance, trainingSet[x], length)
                distances.append((trainingSet[x], dist))
                distances.sort(key=operator.itemgetter(1))
                neighbors = []
        len(distances)
        for x in range(k):
                neighbors.append(distances[x][0])
        return neighbors


# In[18]:


def getResponse(neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
                response = neighbors[x][-1]
                if response == classVotes:
                    classVotes[response] += 1
                else:
                    classVotes[response] = 1
        sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]


# In[19]:


def getscore(x_test, predictions,k):
        test_TP, test_TN, test_FP, test_FN = 0,0,0,0
        #Calculating TP,TN, FP, FN
        for i in range(len(predictions)):
            if(predictions[i] == x_test[i][-1]):
                if(predictions[i] == 0):
                    test_TP += 1
                if(predictions[i] == 1):
                    test_TN += 1
            else:
                if(predictions[i] == 0):
                    test_FP += 1
                if(predictions[i] == 1):
                    test_FN += 1

        accuracy = (float(test_TP + test_TN))/(float(len(predictions)))*100
        precision =  (float(test_TP))/(float(test_TP + test_FP))*100
        recall = (float(test_TP))/(float(test_TP + test_FN))*100
        f_score = (2*precision*recall)/(precision + recall)

        print ('For k = ' + str(k) + '\n')
        print ('Accuracy ' + str(accuracy))
        print('Precision ' +str(precision))
        print('Recall ' + str(recall))
        print('Fscore '+ str(f_score))


# In[21]:


#Executing the main code
def main():
	    # generate predictions
        predictions=[]
        k = 5
        for x in range(len(x_test)):
                neighbors = getNeighbors(x_train, x_test[x], k)
                result = getResponse(neighbors)
                predictions.append(result)
                print('> predicted=' + repr(result) + ', actual=' + repr(x_test[x][-1]))
        getscore(x_test, predictions,k)

main()

