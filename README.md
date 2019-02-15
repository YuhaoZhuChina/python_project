# EE551 python_project
The project is about machine learning.
The description of the project:

Name: The prediction of the quality of red wine based on various machine learning algorithms

1  Problem Statement
   Wine quality measurement is essential in wine industry. It always requires the senses and experience of
   wine tester to determine the quality of wine. Wine tasters can evaluate the wine by the color, texture and
   other appearance characteristics as well as its aroma. Based on the personal decision and judgement, wine
   rating sometimes is not reasonable enough for wine industry . What’s more, wine quality and safety are also
   significient to health. Therefore, except the feedback from wine taster, I also need to find suitable machine
   learning algorithms to build the wine quality classifier and evaluate different kind of wine.

2  Source of Data Set
   The whole data set was downloaded from UCI Machine Learning Repository, it is related to red variants
   of the Portuguese ”Vinho Verde” wine. It contains many sensory attributes such as residual sugar, volatile
   acidity, pH, chlorides, etc. The quality attribute which ranges from 0 to 10 represents the quality of the
   wine from low to high.

3  Implementation Plan for the Project
   I plan to implement 3(maybe more, depends on time) machine learning algorithms to solve this problem:
   (a) Support Vector Machine(SVM):
   MAP represents for “maximum a posterior”. It means that we have to output the hypothesis h in the
   hypothesis space H with the biggest possibility based on the data D which we have observed. It can also
   deal with multi-classification problem and small-size data set. Therefore, it’s suitable for this problem.
   (b) Decision Tree:
   Decision tree is a decision support tool that uses a tree-like graph or model of decisions and their possible
   consequences, including chance event outcomes, resource costs, and utility. It’s also suitable for small-size
   data set and it is easy to build one in python.
   (c) Random Forest:
   The Random Forest is an ensemble learning method for classification, regression and other tasks, that
   operated by constructing a multitude of decision trees at training time and outputting the class that is mode
   of the classes or mean prediction of the individual trees. Random decision forests correct for decision trees’
   overfitting to their training set. It contains several decision trees so it can also solve our problem.

4  Processing the output of different algorithms.
   (a) Compare the accuracy of different machine learning algorithms.
   (b) Test different parameters to improve the accuracy.
   (c) Conclude the advantages and disadvantages of each algorithms which was used in this project.
