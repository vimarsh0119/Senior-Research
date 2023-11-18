pip install scikit-learn

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier

import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # to plot inage, graph
import time

%matplotlib inline

# Load the MNIST dataset from scikit-learn
mnist = fetch_openml('mnist_784')
X, y = mnist.data, mnist.target

X.iloc[:3]

image1 = X.iloc[0]
image1 = list(image1)
image1 = np.array(image1)
image1 = image1.reshape(28,28)

import matplotlib.pyplot as plt
plt.imshow(image1, cmap = plt.cm.gray_r)
plt.axis('off')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = OneVsRestClassifier(KNeighborsClassifier())

# Create a KNN classifier with k=7
knn = KNeighborsClassifier(n_neighbors=7)

# Train the classifier using the training data
knn.fit(X_train, y_train)

# Use the trained KNN classifier to make predictions on the test data
predictions = knn.predict(X_test)

%time
print('KNN Accuracy: %.3f' % accuracy_score(y_test,predictions))

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# to create nice confusion metrics
import seaborn as sns

cm = confusion_matrix(y_test,predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True, fmt='.3f', linewidths=.5, square=True,cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_test,predictions))
plt.title(all_sample_title,size=15)


