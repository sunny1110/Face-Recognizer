#import the necessary packages from the import time


import matplotlib.pyplot as plt
import numpy as np
import time as time

#import the machine learning packages
from sklearn.datasets import fetch_lfw_people
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Downloading the data.
# From the servers and if the data is alredy present.
# In the current working directory load them as numpy array
lfw_people = fetch_lfw_people( min_faces_per_person = 70)


n_samples, h, w = lfw_people.images.shape


#load the data into a variable its target values
# and its target values or say expected values in another variable

X = lfw_people.data # Feature Vector
y = lfw_people.target # Target Variable


n_images = X.shape[0] # Number of Images
n_features = X.shape[1] # Number of Features
person_name = lfw_people.target_names # Name of the person in the images
n_classes = person_name.shape[0]
print "\n"
print person_name
print"\n"

# printing the information about the dataset

print "Total DataSet Size: \n"
print "Number of Images: %d" %n_images
print "Number of Features: %d" %n_features
print "Number of Classes: %d" %n_classes

# Splitting the data set up in a ratio of 7:3, for training, and testing respectively.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)

print "Fitting the classifer to the training set"
t0 = time.time()

logisticreg = LogisticRegression()
logisticreg = logisticreg.fit(X_train, y_train)
y_pred = logisticreg.predict(X_test)
print "Done in %.3f Seconds" %(time.time() - t0)

# Calculate the accuracy of the Current System
num_examples = y_pred.shape[0]
count = 0
for idx in xrange(num_examples):
    if(y_pred[idx] == y_test[idx]):
        count +=1

0
print "Count: %d"%count
print "Examples: %d" %num_examples
accuracy = float(count)/float(num_examples) * 100
print accuracy
