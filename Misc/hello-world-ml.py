"""
Hello World - Machine Learning Recipes #1
Source: https://www.youtube.com/watch?v=cKxRvEZd3Mw&t=165s
"""

from sklearn import tree

features = [[140,1], [130,1], [150,0], [170, 0]]

labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[150,0]]))
