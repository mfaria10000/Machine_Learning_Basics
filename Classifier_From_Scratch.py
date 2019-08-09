from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a, b)



class KNNClassifier():

    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self,x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self,row):
        best_dist = euc(row,self.x_train[0])
        best_index = 0

        for i in range(1, len(self.x_train)):
            dist = euc(row, self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
            return self.y_train[best_index]

from sklearn import datasets
iris = datasets.load_iris()

#here we load the features into X
x = iris.data
#here we load the labels into Y
y = iris.target


#we split the dataset into two group for testing and training
#x_train will have the features and y_train will have the labels, the same for Y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.5)

#ready to train the classifier (we will use our own classifier)
clf = KNNClassifier()

#we use our written function to train the classifier
clf.fit(x_train,y_train)

#now we preditc with the testing data
clf_predic = clf.predict(x_test)

print(clf_predic)

#lets print the accuracy of the predictions from sklearn
from sklearn.metrics import accuracy_score
#print(len(y_test))
#print(len(clf_predic))
print(accuracy_score(y_test,clf_predic))