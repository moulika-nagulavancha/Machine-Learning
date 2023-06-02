# Predict the Flowers (Iris Virginica) using IRIS dataset
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
iris = datasets.load_iris()
# ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']
# print(iris.keys())
# print(iris.data)
# print(iris.target)
# print(iris.DESCR)
# print(iris.data.shape)

iris_X = iris.data[:, 3:]
# print(iris_X)
#  make a binary classifier based on the data whether the flower is Iris Virginica (2 is the label) or not
iris_Y = (iris.target == 2).astype(int)
# print(iris_Y)

# Train a Logistic classifier
classifier = LogisticRegression()
# Feed the data to the classifier
classifier.fit(iris_X, iris_Y)

# example = classifier.predict([[1.6]])
example = classifier.predict([[2.6]])

print(example)

# Use matplotlib to plot the visual graph
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
# Gives the actual probability value within the X_new range
Y_probability = classifier.predict_proba(X_new)
# print(Y_probability)
# plot the graph
plt.plot(X_new, Y_probability[:, 1], 'g-', label='virginca', linewidth=2)
plt.grid(True)
plt.title('Probability of the Flower Iris Virginica')
plt.legend()
plt.show()