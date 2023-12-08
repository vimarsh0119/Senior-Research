# Image Recognition Using The MNIST Dataset of Handwritten Digits

It is important to study and use technology such as cybersecurity and artificial intelligence so that effective strategies of keeping information secure can be made. One of the most useful ways to secure information on mobile devices can be biometric authentication. Some biometric software that are being used today include facial recognition and image recognition. The objective of the research project is to test the k-NN (K-Nearest Neighbor) algorithm's ability to classify any image accurately as opposed to merely recognizing faces for image recognition. The K-Nearest Neighbors (KNN) algorithm is a simple but efficient approach for classification and regression tasks. The algorithm is also simple to learn and is flexible. Using the KNN algorithm could help perform facial recognition and image recognition. The KNN algorithm could also provide incredibly accurate classifications and detections which could compete with the most accurate models. KNN could therefore be used for a variety of mobile applications that can strengthen security.






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

Output:
<img width="998" alt="Screenshot 2023-11-17 at 9 15 55 PM" src="https://github.com/vimarsh0119/Senior-Research/assets/149597902/64842fb5-d8ac-455b-b6de-45601ebe5e05">

3 rows × 784 columns

image1 = X.iloc[0]
image1 = list(image1)
image1 = np.array(image1)
image1 = image1.reshape(28,28)

import matplotlib.pyplot as plt
plt.imshow(image1, cmap = plt.cm.gray_r)
plt.axis('off')

Output:
(-0.5, 27.5, 27.5, -0.5)

![image](https://github.com/vimarsh0119/Senior-Research/assets/149597902/f0373560-8c78-4819-bcc6-7c3f5ca7335f)

import matplotlib.pyplot as plt

# set up the plot
figure, axes = plt.subplots(2, 5, figsize=(28, 28))

for ax, image, number in zip(axes.ravel(), X.values, y):
    ax.axis('off')
    ax.imshow(image.reshape((28, 28)), cmap=plt.cm.gray_r)

plt.show()

Output:
![image](https://github.com/vimarsh0119/Senior-Research/assets/149597902/f661a8d2-e24f-4f14-88b7-a4574a80d098)

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

Output:
CPU times: user 4 µs, sys: 1 µs, total: 5 µs
Wall time: 7.87 µs

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

Output:
Accuracy: 0.9687142857142857

# To create nice confusion metrics
import seaborn as sns

cm = confusion_matrix(y_test,predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True, fmt='.3f', linewidths=.5, square=True,cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_test,predictions))
plt.title(all_sample_title,size=15)
KNN Accuracy: 0.969

Output:
Text(0.5, 1.0, 'Accuracy Score: 0.9687142857142857')

![image](https://github.com/vimarsh0119/Senior-Research/assets/149597902/2dff9339-0173-4424-9a14-70e3ee92aab5)
