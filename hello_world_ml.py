#importing tree from sklearn library
from sklearn import tree
#features for learning
features = [[140,1],[130,1],[150,0],[170,0]]
#lables for learning
lables = [0,0,1,1]
#defining a classifier
clf = tree.DecisionTreeClassifier()
#fitting the data for future prediction
clf = clf.fit(features, lables)
#predicting from input.
print(clf.predict([[10,0]]))