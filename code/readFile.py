#!/usr/bin/env python
# coding: utf-8

# # Read data from csv file

# In[1]:


import csv
import numpy
#open the csv document.
data=open("winequality-red.csv")


#read the first row of data which are attributes into list names, then read the rest data into list row, 
#meanwhile, extract the last column of list row, then, put it in the list labels.
xlist = []
labels = []
names = []
firstline = True
for line in data:
    if firstline:
        names = line.strip().split(';')
        firstline = False
    else:
        row = line.strip().split(';')
        labels.append(float(row[-1]))
        row.pop()
        floatrow = [float(num) for num in row]
        xlist.append(floatrow)
#transfer into numpy format.
x = numpy.array(xlist)
y = numpy.array(labels)
winenames = numpy.array(names)

