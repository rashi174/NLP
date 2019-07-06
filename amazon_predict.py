###python code to predict whether the review is negative or positive using nlp(natural language processing)


##let's begin by importing libraries


import pandas as pd

##reading the text file with tab spaced values
dataset=pd.read_csv('amazon_cells_labelled.txt',delimiter='\t',header=None)

##naming the coloums as 'review' and 'rating'
dataset.columns = ['Review', 'rating']

import re  ##regular expression
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


from nltk.stem.porter import PorterStemmer
#from nltk.stem.wordnet import WordNetLemmatizer 

##creating an empty list
corpus = []

##loop will run for 1000 rows 0 to 999
#with regular express we will eliminate unneccesary charaters like number or special symbols
##then convert the string in lowe case
#finally split it to make a list
##remove stop words(is ,am ,are, the etc..)


for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    
    #lem = WordNetLemmatizer() #Another way of finding root word
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    #review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
features = cv.fit_transform(corpus).toarray()
labels = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = classifier.predict(features_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(labels_test, labels_pred)
"""

#applying knn on this text dataset
# Fitting Knn to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = classifier.predict(features_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(labels_test, labels_pred)

"""
##both alogorithms are giving almost same score.
##score is not so good bcoz data is not large
    
