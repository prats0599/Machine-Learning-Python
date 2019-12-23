# Natural Language Processing

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords') # to remove all unneccesary words such as a an the this etc
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):    
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    # set function is used so that algo can execute faster(rather than using lists). 
    # Stemming (taking root of words eg love loved loving == love so that we dont need to keep different versions of the same word)
    # done in order to reduce sparsity
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
# creating a sparse matrix(X). the model is essentialy to clean all the reviews so as to simplify it and try to reduce the number of words.
# ..through the process of tokenization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
    
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# fitting classifier to the training set
from sklearn.naive_bayes import GaussianNB 
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# predict the test set results
y_pred = classifier.predict(X_test)

# making the confusion matrix(contains correct and incorrect predictions made by our model)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)   #will be a vector of size 100.
# accuracy: (55+91)/200 = 73%


