
from  sklearn import tree
#this will be input to the classifier
#features = [[140,'smooth'],[130,'smooth'],[150,'bumpy'],[170,'bumpy']]
#this will be output to the classifier
#labels = [['apples'],['apples'],['oranges'],['oranges']]


#we will substitute the labels for numbers
features = [[140,0],[130,0],[150,1],[170,1]]
labels = [[0],[0],[1],[1]]

#we will build a decision tree classifier
clf = tree.DecisionTreeClassifier()

#we will use a training algorithm called fit
cls = clf.fit(features,labels)
# at this point we have a trained classifier, lets use it to predict a fruit
#here we pass 150 for the weight and 0 for bumpy
print (clf.predict([[150,0]]))



