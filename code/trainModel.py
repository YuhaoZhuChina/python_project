#!/usr/bin/env python
# coding: utf-8

# # Train the random forest and output the prediction accuracy 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import pylab as plot

#randomly draw 30% data from the dataset for testing, and set the random_state equals 531.
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.30,random_state=531)

mseoos = []
accuracy = []
#test the accruacy in terms of amounts of trees which range from 50 to 500, and the single step is 10.
ntreelist=range(50,500,10)
dic = {}
for itrees in ntreelist:
    count=0
    depth=None
    maxfeat=4
    #train the Random Forest model.
    winerandomforestmodel=ensemble.RandomForestClassifier(n_estimators=itrees,max_depth=depth,max_features=maxfeat,oob_score=False,random_state=531)
    winerandomforestmodel.fit(xtrain,ytrain)
    prediction=winerandomforestmodel.predict(xtest)
    #put the prediction accuracy into dictionary.
    for i in range(len(prediction)):
        if prediction[i]==ytest[i]:
            count += 1
    accuracy.append((count/len(prediction)))
    dic[itrees] = count/len(prediction)
    mseoos.append(mean_squared_error(ytest,prediction))
for key,value in dic.items():
    if value == max(dic.values()):
        print('The maximum acurracy is: '+ str(key) +'==>'+str(value))
print("MSE: " + str(mseoos[-1]))
plot.plot(ntreelist,accuracy)
plot.xlabel("Number Of Trees")
plot.ylabel("Accuracy")
plot.show()

