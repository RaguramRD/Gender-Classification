#!/usr/bin/env python
# coding: utf-8
'''
# # FIT5149 - Applied Data Analysis
# 
# ## Assignment 02
# 
# ## Group 02
# 
# ### Group member 1: Raguram Ramakrishnasamy Dhandapani , Student ID: 30151325
# ### Group member 2: Thirugnanam Ramanathan, Student ID: 30404975
# ### Group member 3: Nuwan Chamila Withana Gamage, Student ID: 29255066
# 
'''

### Libraries Required



#Importing necessary lib
import re
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import RegexpTokenizer 
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


#Cross validation function
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#Feature Selection
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification



### Input Data
train_data = pd.read_csv('train_labels.csv')
print('Train data')
print(train_data.head())

### Test Data
test_data = pd.read_csv('test_labels.csv')
print('\n Test data')
print(test_data.head())


'''
Extracting the text from the Train and test dataset
-Each document is opened as a file and read using readlines funtion
- Since the file is in XML format, regex is used to extract the information needed
- Then each document is stored in a list for analysis
- Separate list are created for Train and test data
'''

### Extracting data from XML files

#function for extracting the data
def extract_xml_data(corpus):
    
    text_extract = re.search('<documents>(.*)</documents>',str(corpus), flags=16)  # search for content with documents tag
    
    #enters only if the extracted word is not null
    if text_extract != '':
        text_list = re.finditer('<documents.*?>(.*?)</documents>',text_extract.group(0), flags=16)
        
        #looped for all items in the list
        for item in text_list:
            text = re.sub('<documents*>','',item.group(0))    # opening document tag is substituted to null
            text = re.sub('</documents*>','',text)            # closing document tag is substituted to null
            text = re.sub('<!\[CDATA\[','',text)              # common term on twitter is substituted to null
            text = re.sub('http.*>','',text)                  # hyperlink is substituted to null
            text = re.sub('[^a-zA-Z\s]','',text)              # regex expression is used to limit the text
            
    return(text)


##TRAIN DATA
#generating an empty list for data after extraction
corpus_text = []

# loop runs each id(tweet) - 3600 xml files (3100 train and 500 test)
for id in list(train_data.iloc[:,0]):
    corpus = open('data/'+id+'.xml','r',encoding="utf8")
    text = extract_xml_data(corpus.read())
    corpus_text.append(text)

##TEST DATA
#empty list for extracted test data 
corpus_text_test = []

#loop for every test xml file
for id in list(test_data.iloc[:,0]):
    corpus = open('data/'+id+'.xml','r',encoding="utf8")
    text = extract_xml_data(corpus.read())
    corpus_text_test.append(text)




### Total entries in each dataset
## Looking at the labels ratio - train set
print('\nNo of tweets by Male (out of 3100) - ',train_data.gender[train_data.gender=='male'].count())
print('No of tweets by Female (out of 3100) - ',train_data.gender[train_data.gender=='female'].count())

## Looking at the labels ratio - test set
print('\nNo of tweets by Male (out of 500) - ',test_data.gender[train_data.gender=='male'].count())
print('No of tweets by Female (out of 500) - ',test_data.gender[train_data.gender=='female'].count())



'''
Processing the Train and test dataset
- Basic Text processing done on the train data
- Tokenisation using Regexp
- Case normalisation, N-gram extraction, Stop words removal
- Lemmatisation done with WordNet Lemmatisatiser 
'''

### Text preprocessing

#lemmatizer class with regex tokenizer
class token_lemm(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self,doc):
        return [self.wnl.lemmatize(t) for t in RegexpTokenizer('[a-zA-Z]+').tokenize(doc)]


''' Only alphabets is taken in our model for definng the label variable, So [a-zA-Z]+ allows all alphabets only.'''

#TfidfVectorizer is used to vectoriser 
vectorizer=TfidfVectorizer(analyzer='word',input='content',
                                lowercase=True, ngram_range=(1,3),    # monogram, bigram, tri grams are taken into consideration
                                min_df=0.05,max_df=0.95,              # document range taken as 5% to 95%
                                stop_words=set(list(string.punctuation) + stopwords.words('english')),#punctuation and stopwords are removed
                                tokenizer = token_lemm())             # lemmatizer function is called.




## TRAIN DATA
#vectorizer is built on the cleaned text
train_fit = vectorizer.fit_transform(corpus_text)

#Visualizing data of the trained model
print('\nTFIDF matrix results in '+ str(train_fit.shape[1]) + ' features')

#Train Labels
labels = list(pd.factorize(list(train_data.iloc[:,1] ),sort=True)[0])


## TEST DATA
#transforming the test data using the model build in 
test_fit = vectorizer.transform(corpus_text_test)


#test labels
test_labels = list(pd.factorize( list(test_data.iloc[:,1]) ,sort=True)[0])



'''
After fine tuning all the above models as mentioned in the report, the best model is fitted below for the data
Logistic Regression
'''

## Logistic Regression
print('\n\nFitting Logistic Regression')
final_model = LogisticRegression(C = 1.52642, solver = 'newton-cg')
final_model.fit(train_fit, labels)
final_pred = final_model.predict(test_fit)
print('Accuracy for the Model is ' + str(accuracy_score(test_labels,final_pred)*100)+ '%')



### Storing test data
final_df = pd.DataFrame(columns=['id', 'gender'])

#Changing 1/0 to male/female
final_pred = list(final_pred)
for i in range(len(final_pred)):
	if final_pred[i] == 1:
		final_pred[i] = 'male'
	else:
		final_pred[i] = 'female'
	   
final_df['id'] = test_data.iloc[:,0]
final_df['gender'] = final_pred


final_df.to_csv('pred_labels.csv', index=False)


## References
''' https://www.sciencedirect.com/science/article/pii/S1877050916326849
    https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
    https://realpython.com/python-keras-text-classification/
    https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769
    https://scikit-learn.org/stable/supervised_learning.html
    https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py '''
