from sklearn import datasets
iris = datasets.load_iris()


#here we load the features into X
x = iris.data
#here we load the labels into Y
y = iris.target

from sklearn.model_selection import train_test_split
#we split the dataset into two group for testing and training
#x_train will have the features and y_train will have the labels, the same for Y
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.5)

#ready to train the classifier (CLASSIFIER 1)
#from sklearn import tree
#clf = tree.DecisionTreeClassifier()

#ready to train the classifier (CLASSIFIER 2)
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()

clf.fit(x_train,y_train)
clf_predic = clf.predict(x_test)
print(clf_predic)

#here we test git changes
#lets print the accuracy of the predictions from sklearn
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,clf_predic))