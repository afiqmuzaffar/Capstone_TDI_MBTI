import streamlit as st
import numpy as np
import pandas as pd

st.title('Looking at the big picture:')

st.write('''In this project, a natural language processing (NLP) model is developed to predict person's personality type using Myers Briggs Personality test (MBTI). The input to the model is people's sample text and the model makes predictions about their MBTI's personality type.

The Myers–Briggs Type Indicator (MBTI) is an introspective self-report questionnaire indicating differing psychological preferences in how people perceive the world and make decisions. The test attempts to assign four categories: introversion or extroversion, sensing or intuition, thinking or feeling, judging or perceiving. One letter from each category is taken to produce a four-letter test result, like "INFJ" or "ENFP". (Source: Wikipedia)

This project has wide range of business applications. If we know how certain people interact with their environment we can make proper recommendations to them. The recommendations can range from book, or movie recommendations to the next travel destination. For example, introverted people may want to be engaged in more relaxed activities in comparison to extroverts who like large parties and adventures.''')

st.title('Getting a data:')

st.write('''This project has four parts which look at different aspects of a person's personality. This notebook is looking at predicting "introversion" in people by analyzing their social media posts. Dataset for this project is mainly from Kaggle website complemented by tweets consisting of personality tags. Dataset: https://www.kaggle.com/datasnaek/mbti-type The web scraping algorithm to acquire tweets is reflected at the end of this notebook in appendix.''')


dataset = pd.read_csv('mbti.csv')

st.write(dataset.head(10))

D=dataset.iloc[:]
X = D.iloc[:, :-1].values
Y = D.iloc[:, -1].values
from collections import Counter
my_dt = pd.DataFrame(X)
Lx = [i[0] for i in my_dt.values.tolist()]
result = {}    
for word in Lx:                                                                                                                                                                                               
    result[word] = result.get(word, 0) + 1    

D={k: v for k, v in sorted(result.items(), key=lambda item: item[1])}

PD=pd.Series(D).to_frame('A')
st.bar_chart(PD)

st.title('Feature engineering:')
st.write(''' There could be two approaches to solve the problem: 1. Classifying each personality type among 16 different classes 2. Using multi-label classification and classifying between four different\
         aspects of the personality. Second approach is selected to avoid the possible issues with the class imbalance and also the better accuracy. Four different labels are between 'I' or 'E', 'N' or 'S',\
             'F' or 'T', 'J' or 'P' in which 'I' stands for introvert, 'E' stands for extrovert, 'N' is intuitive, 'S' is sensing, 'F' is feeling, 'T' is thinking, 'P' is perceiving, and 'J' is judging.
''')

st.write('''The first step is to categorize the training and test sets. Categorization is to split data based on following labels: 'I' or 'E', 'N' or 'S',\
             'F' or 'T', 'J' or 'P'  ''')

#storing all the text
I1=[]
I2=[]
I3=[]
I4=[]
I5=[]
N=0
Int_cloud=[]
Ext_cloud=[]
import re
for i in range (0,len(Y)):
    I1.append(Y[i])
    
    if X[i][0][0]=='I':
        I2.append("I")
        YY= re.sub(r'\b\w{1,3}\b', '', Y[i])
        YY= re.sub("[^a-zA-Z]"," ", YY)
        YY= re.sub("https"," ", YY)
        YY= re.sub("infp"," ", YY)
        YY= re.sub("doesn"," ", YY)
        YY= re.sub("didn"," ", YY)
        Int_cloud.append(YY)
    
    if X[i][0][0]=='E':
        I2.append("E")
        YY= re.sub(r'\b\w{1,3}\b', '', Y[i])
        YY= re.sub("[^a-zA-Z]"," ", YY)
        YY= re.sub("https"," ", YY)
        YY= re.sub("enfj"," ", YY)
        YY= re.sub("entj"," ", YY)
        YY= re.sub("infp"," ", YY)
        YY= re.sub("doesn"," ", YY)
        YY= re.sub("didn"," ", YY)
        Ext_cloud.append(YY)
        
    if X[i][0][1]=='N':
        I3.append("N")
    
    if X[i][0][1]=='S':
        I3.append("S")  
        
    if X[i][0][2]=='F':
        I4.append("F")
    
    if X[i][0][2]=='T':
        I4.append("T")          
        
    if X[i][0][3]=='P':                   
        I5.append("P")
    
    if X[i][0][3]=='J':
        I5.append("J")

p1 = pd.Series([item for item in I1])
p2 = pd.Series([item for item in I2])
p3 = pd.Series([item for item in I3])
p4 = pd.Series([item for item in I4])
p5 = pd.Series([item for item in I5])

II=pd.DataFrame({"words":p1,"Intorvert":p2,"Intuit":p3,"Feeling":p4,"Precieve":p5})

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([str(Int_cloud), str(Ext_cloud)])
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
TFIDF_PERS=pd.DataFrame({"I":df.iloc[0],"E":df.iloc[1]})

I=pd.DataFrame({"I":TFIDF_PERS["I"]}).sort_values('I',ascending=False)
E=pd.DataFrame({"E":TFIDF_PERS["E"]}).sort_values('E',ascending=False)
st.write('''This section explores the word frequency (TFIDF) for the introverted and extroverted people. Word clouds can provide a nice visualization of more frequent words in the text.\
         Following two figures are more or less the 100 most important words for introverts and extroverts.''')
C=[]
for i in range(100,200):
    C.append(I.index[i])
C1=str(C)
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
import pandas as pd 
import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm')
#ls={"infp",'intj','intp','infj'}
#STOP_WORDS.add(ls)
STOP_WORDS |= {"infp",'intj','intp','infj',"enfp",'entj','entp','enfj',\
               "isfp",'istj','istp','isfj',"esfp",'estj','estp','esfj',\
                   'https'}
stop_words = spacy.lang.en.stop_words.STOP_WORDS  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stop_words, 
                min_font_size = 5).generate(C1) 
  
#plot the WordCloud image                        
plt.figure(figsize = (6, 6), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
st.set_option('deprecation.showPyplotGlobalUse', False)  
plt.savefig("Introvert.png")
st.header("Important words used by introverts") 
st.image("Introvert.png", use_column_width=True)

st.write('''The word "little" is an important word used in both categories.\
         The first 100 words are pretty much equally used by both introverts and extroverts, however discrepancies appear for more-frequent words after 100. \
         It is seen that extroverts use  words such as "long" , "guys" compared to introverts who use "kind", "usually".''')

CC=[]
for i in range(100,200):
    CC.append(E.index[i])
C2=str(CC)


wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stop_words, 
                min_font_size = 5).generate(C2) 
# plot the WordCloud image                        
plt.figure(figsize = (6, 6), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.savefig("Extrovert.png")
st.set_option('deprecation.showPyplotGlobalUse', False) 
st.header("Important words used by extroverts") 
st.image("Extrovert.png", use_column_width=True)

st.title('Test preprocessing:')
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
# Custom transformer using spaCy
# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    sentence= re.sub("[^a-zA-Z]"," ", sentence)
    sentence= re.sub(r'\b\w{1,3}\b', '', sentence)
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
punctuations = string.punctuation

def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

st.title("Training the Machine Learning Model:")
st.write("Logistic reggression...")

import numpy as np
X_train=II["words"]
y_multilabel=np.c_[II["Intorvert"],II["Intuit"],II["Feeling"],II["Precieve"]]

from sklearn.preprocessing import OrdinalEncoder
Inp=y_multilabel

ordinal_encoder = OrdinalEncoder()
Iencd = ordinal_encoder.fit_transform(Inp)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train,Iencd, test_size = 0.2, random_state = 42)

import re
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
#Training the model using parameters from the grid search
classifier = LogisticRegression(penalty='l2',dual=False, tol=0.1, C=0.01, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=10,solver='liblinear', max_iter=500, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', OneVsRestClassifier(classifier))])

# model generation
pipe.fit(X_train,y_train)

from sklearn.metrics import precision_score
predicted1 = pipe.predict(X_test)
precision_score(y_test, predicted1, average='macro')

from sklearn import metrics
# Predicting with a test dataset
predicted1 = pipe.predict(X_test)
#sklearn.metrics.f1_score
print("logestic Accuracy:",metrics.f1_score(y_test, predicted1, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn'))
import pickle
pickle.dump(pipe, open('full_model.pkl', 'wb'))
load_clf = pickle.load(open('full_model.pkl', 'rb'))
import dill
import gzip
with gzip.open('Fullmodel.dill.gzip', 'wb') as f:
    dill.dump(pipe, f, recurse=True)
    '''
    
st.title("Examining the Full Model:")

def ID(predict):
    IE=[]
    NS=[]
    FT=[]
    PJ=[]
    L=[]
    for i in range (len(predict)):
        if predict[i][0]==1:          
            IE.append("I")
        else:
            IE.append("E")
        if predict[i][1]==1:
            NS.append("N")
        else:
            NS.append("S")
        if predict[i][2]==1:
            FT.append("F")
        else:
            FT.append("T")
        if predict[i][3]==1:
            PJ.append("P")
        else:
            PJ.append("J")
        L.append(IE[i]+NS[i]+ FT[i]+PJ[i])
    return L

St=str()
IEt=[]
NSt=[]
FTt=[]
PJt=[]
for j in range (len(y_test)):
    if y_test[j][0]==1:
        St="I"
        IEt.append(St)
    else:
        St="E"
        IEt.append(St)
        
    if y_test[j][1]==1:
        St="N"
        NSt.append(St)
    else:
        St="S"
        NSt.append(St)
        
    if y_test[j][2]==1:
        St="F"
        FTt.append(St)
    else:
        St="T"
        FTt.append(St)
        
    if y_test[j][3]==1:
        St="P"
        PJt.append(St)
    else:
        St="J"
        PJt.append(St)
PERSt=[]
St=str()
for i in range(len(IEt)):
    St=IEt[i]+NSt[i]+FTt[i]+PJt[i]
    PERSt.append(St) 
PERS=ID(predicted1)
dictionary = dict(zip(PERS, PERSt))

st.write(dictionary)

# Creation of the actual interface, using authentication
import tweepy
consumer_key='nKzFqkTzhT697GAsiW16TlkUc'
consumer_secret='3dnxrzKtcG5BjXmX12lg7dySf99J9x23dn6PTjnPBwOYQsEXJz'

auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

# Creation of the actual interface, using authentication
api = tweepy.API(auth)
Txx=[]
for status in tweepy.Cursor(api.user_timeline, screen_name='@JoeBiden', tweet_mode="extended").items(1000):
    Txx.append(status.full_text)

d={}
L=[]
s=str()
import spacy
nlp = spacy.load("en_core_web_sm")
for t in Txx:   
    doc = nlp(t)
    for sent in doc.sents:
        if sent[0].is_title and sent[-1].is_punct:
            has_noun = 2
            has_verb = 1
            for token in sent:
                if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                    has_noun -= 1
                elif token.pos_ == "VERB":
                    has_verb -= 1
            if has_noun < 1 and has_verb < 1:
                 L.append(sent.string.strip())

NP=pipe.predict(L)

NM=np.mean(NP, axis=0)

import altair as alt

source = alt.pd.DataFrame([
      {
        "question": "I/E",
        "type": "Introversion",
        "value": NM[0],
        "percentage": 90,
        "percentage_start": 0,
        "percentage_end": 97
      },
      {
        "question": "N/S",
        "type": "Intuition",
        "value":  NM[1],
        "percentage": 0.1,
        "percentage_start": 0,
        "percentage_end": 1
      },
      {
        "question": "F/T",
        "type": "Feeling",
        "value": NM[2],
        "percentage": 90,
        "percentage_start": 0,
        "percentage_end": 94
      },
      {
        "question": "P/J",
        "type": "Perceiving",
        "value": NM[3],
        "percentage": 90,
        "percentage_start": 0,
        "percentage_end": 92
      },
     
     
])

color_scale = alt.Scale(
    domain=[
        "Introversion",
        "Intuition",
        "Feeling",
        "Perceiving"
    ],
    range=["#c30d24", "#f3a583", "#cccccc", "#94c6da"]
)

y_axis = alt.Axis(
    title="Joe Biden's MBTI",
    offset=5,
    ticks=False,
    minExtent=60,
    domain=False
)

chart=alt.Chart(source).mark_bar().encode(
    x='percentage_start:Q',
    x2='percentage_end:Q',
    y=alt.Y('question:N', axis=y_axis),
    color=alt.Color(
        'type:N',
        legend=alt.Legend( title='Aspects of personality'),
        scale=color_scale,
    )
)'''
