
import numpy as np
from  sklearn import tree
from sklearn.datasets import load_iris
iris = load_iris()


#you can look ate the feature and target labels
print('######## FEATURES AND LABELS #######')
print (iris.feature_names, iris.target_names)

#working with the dataset
#target contains the labels and is a one dimentional array
#print (iris.target_names[0])
#print(iris.target[0])

#data contains the features and is a two dimentional array
#print(iris.data[0])

#for i in range(len(iris.target)):
#    print ('Example %d: Label %s: Features %s' % (i, iris.target[i], iris.data[i]))

#Now we are going to separate the testing data from the training data
# the dataset is ordered so the first setosa is at 0, the first versicolor is at 50 and the first virginica is at 100
test_idx = [0,50,100]

# lets remove the three entries as our training data and leave the majority of the data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#creating the testing data with just the three the examples that we removed from the dataset
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#lets train our classifier with the training data (majority of the data)
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print('######## SAMPLE DATA - BEGIN #######')
print(test_data)
print('######## SAMPLE DATA - END #######')


#now that we have trained our classifier lets print the three labels that we have remove for testing
print('######## EXPECTED RESULT - BEGIN #######')
print(test_target)
print('######## EXPECTED RESULT - END #######')

#now lets predict the result based on our testing data
#here we will pass the futures (DATA FOR THE 3 FLOWERS) that we separated and get back as a result the labels
#This should give  3 answers in a array
print('######## PREDICTION RESULT - BEGIN #######')
lst_clf = clf.predict(test_data)
print(lst_clf)
print('######## PREDICTION RESULT - END #######')

print('######## LABELED RESULT - END #######')
#first loop uses the value 0, 1 or 2 to get the name label of the flower
for clf_item in lst_clf:
    print(iris.target_names[clf_item])
    #second loop gets the features value array of each flower and print the label for the future from the future_names array
    for i in test_data[clf_item]:
        #here we retrieve the tuple index 0 that represents the index of one of the 4 values of the futures and use it as index to get the labels
        var_ndx = int(np.where(test_data[clf_item] == i)[0])
        print(iris.feature_names[var_ndx] + ':' + str(i))

print('######## LABELED RESULT - END #######')








