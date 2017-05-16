"""
A simple demo predicting whether someone is male or female based on height, weight and shoe size

Source: https://www.youtube.com/watch?v=T5pRlIbr6gg&index=1&list=PL2-dafEMk2A6QKz1mrk1uIGfHkC1zZ6UU
"""

from sklearn import tree

# [height, weight, size]

X = [[181, 80, 44],
      [177, 70, 40],
      [160, 60, 38],
      [154, 54, 37],
      [130, 60, 42],
      [150, 52, 37],
      [175, 65. 43]
    ]

Y = ['male', 'male', 'male', 'female', 'female', 'femmale', 'male']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X, Y)

prediction = clf.predict([[190, 70, 43]])

print(prediction)
