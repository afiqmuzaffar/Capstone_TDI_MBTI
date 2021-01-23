# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 11:45:34 2020

@author: fsaff
"""

import streamlit as st
import numpy as np
import pandas as pd
import dill
import gzip



import sklearn
st.image("fig.png", use_column_width=True)

st.title('Looking at the big picture:')


st.write('''In this project, a natural language processing (NLP) model is developed to predict person's personality type using Myers Briggs Personality test (MBTI). The input to the model is people's sample text and the model makes predictions about their MBTI's personality type.

The Myers–Briggs Type Indicator (MBTI) is an introspective self-report questionnaire indicating differing psychological preferences in how people perceive the world and make decisions. The test attempts to assign four categories: introversion or extroversion, sensing or intuition, thinking or feeling, judging or perceiving. One letter from each category is taken to produce a four-letter test result, like "INFJ" or "ENFP". (Source: Wikipedia)

This project has wide range of business applications. If we know how certain people interact with their environment we can make proper recommendations to them. The recommendations can range from book, or movie recommendations to the next travel destination. For example, introverted people may want to be engaged in more relaxed activities in comparison to extroverts who like large parties and adventures.''')

st.title('Getting a data:')

st.write('''This project has four parts which look at different aspects of a person's personality. This notebook is looking at predicting four different aspects of personality in people by analyzing their social media posts. Dataset for this project is mainly from Kaggle website complemented by tweets consisting of personality tags. Dataset: https://www.kaggle.com/datasnaek/mbti-type. The accuracy of the model is 80% on a Test set.''')


st.image("fig.1.png", use_column_width=True)

st.title('Feature engineering:')
st.write(''' There could be two approaches to solve the problem: 1. Classifying each personality type among 16 different classes 2. Using multi-label classification and classifying between four different\
         aspects of the personality. Second approach is selected to avoid the possible issues with the class imbalance and also the better accuracy. Four different labels are between 'I' or 'E', 'N' or 'S',\
             'F' or 'T', 'J' or 'P' in which 'I' stands for introvert, 'E' stands for extrovert, 'N' is intuitive, 'S' is sensing, 'F' is feeling, 'T' is thinking, 'P' is perceiving, and 'J' is judging.
''')

st.write('''The first step is to categorize the training and test sets. Categorization is to split data based on following labels: 'I' or 'E', 'N' or 'S',\
             'F' or 'T', 'J' or 'P'  ''')
st.image("Introvert.png", use_column_width=True)

st.write('''The word "little" is an important word used in both categories.\
         The first 100 words are pretty much equally used by both introverts and extroverts, however discrepancies appear for more-frequent words after 100. \
         It is seen that extroverts use  words such as "long" , "guys" compared to introverts who use "kind", "usually".''')

st.image("Extrovert.png", use_column_width=True)


dill._dill._reverse_typemap['ClassType'] = type
with gzip.open('Fullmodel.dill.gzip', 'rb') as f2:
    model2= dill.load(f2)



    
#load_clf = pickle.load(open('full_model.pkl', 'rb'))
st.title("Examining the Full Model:")

option = st.selectbox(
     'Which book you would like to analyze?',
     ('Harry_Potter_and_the_Chamber_of_Secrets', 'Kafka_on_the_Shore', 'The_Wind_Up_Bird_Chronicle', 'educated', 'Man_s_Search_for_Meaning','The Selfish Gene','Funny in Farsi: A Memoir of Growing Up Iranian in America'\
        ,'Escape_from_Freedom', 'being-mortal','Sapiens: A Brief History of Humankind','On_Liberty','how-to-change-your-mind','The Course of Love','When Nietzsche Wept: A Novel Of Obsession','The Little Prince'))
if option=='Harry_Potter_and_the_Chamber_of_Secrets':
    url="https://www.goodreads.com/book/show/15881.Harry_Potter_and_the_Chamber_of_Secrets?ac=1&from_search=true&qid=XmkTPrJ2OK&rank=5"
if option=='Kafka_on_the_Shore':
    url='https://www.goodreads.com/book/show/4929.Kafka_on_the_Shore?ac=1&from_search=true&qid=qeJwL6gJes&rank=1'
if option=='The_Wind_Up_Bird_Chronicle':
    url='https://www.goodreads.com/book/show/11275.The_Wind_Up_Bird_Chronicle?ac=1&from_search=true&qid=eOOLtwut3h&rank=4'
if option=='educated':
    url='https://www.goodreads.com/book/show/35133922-educated?ac=1&from_search=true&qid=ElxZvyPDIg&rank=1'
if option=='Man_s_Search_for_Meaning':
    url='https://www.goodreads.com/book/show/4069.Man_s_Search_for_Meaning?ac=1&from_search=true&qid=kWZRXUbasg&rank=1'
if option=='The Selfish Gene':
    url='https://www.goodreads.com/book/show/61535.The_Selfish_Gene?ac=1&from_search=true&qid=II2JLWEiaO&rank=1'
if option=='Funny in Farsi: A Memoir of Growing Up Iranian in America':
    url='https://www.goodreads.com/search?q=Funny+in+Farsi%3A+A+Memoir+of+Growing+Up+Iranian+in+America&qid=xDqonEFTcb'
if option=='Escape_from_Freedom':
    url='https://www.goodreads.com/book/show/25491.Escape_from_Freedom?from_search=true&from_srp=true&qid=fgNunQX52H&rank=1'
if option=='being-mortal':
    url='https://www.goodreads.com/book/show/20696006-being-mortal?from_search=true&from_srp=true&qid=lgzn0HTtZv&rank=11'
if option=='Sapiens: A Brief History of Humankind':
    url='https://www.goodreads.com/book/show/23692271-sapiens?ac=1&from_search=true&qid=2L7H4JUNE2&rank=1'
if option=='On_Liberty':
    url='https://www.goodreads.com/book/show/385228.On_Liberty?from_search=true&from_srp=true&qid=nxKQ9To9JP&rank=1'
if option=='how-to-change-your-mind':
    url='https://www.goodreads.com/book/show/36613747-how-to-change-your-mind?ac=1&from_search=true&qid=bimaCYWg52&rank=1'
if option=='The Course of Love':
    url='https://www.goodreads.com/book/show/27845690-the-course-of-love?ac=1&from_search=true&qid=aSstjm0UTl&rank=4'
if option=='When Nietzsche Wept: A Novel Of Obsession':
    url='https://www.goodreads.com/book/show/18981050-when-nietzsche-wept?from_search=true&from_srp=true&qid=e1nEEIhFzZ&rank=1'    
if option=='The Little Prince':
    url='https://www.goodreads.com/book/show/8841.The_Tale_of_the_Rose?ac=1&from_search=true&qid=tgaJnLQMy9&rank=3'    
    
from sklearn import metrics
import requests
from bs4 import BeautifulSoup 
st.write('Sample reveiw by goodreads reviewer:')
response = requests.get(url).text
bs = BeautifulSoup(response, "html.parser")
Link_date=bs.select(".readable")
Link_date[1].text
B=[]
for l in Link_date:
    B.append(l.text)
s=[B for B in B if len(B)>100]

predicted2 = model2.predict(s)

NM=np.mean(predicted2, axis=0)

if NM[0]>0.5:
    st.write('The readers of {} are {}% INTROVERT type'. format(option,round(NM[0]*100)))
else:
    st.write('The reasders of {} are  {}% EXTOVERT type'. format(option,round(100-NM[0]*100)))

if NM[1]>0.5:
    st.write('The readers of {} are {}% INTUITIVE type'. format(option,round(NM[1]*100)))
else:
    st.write('The readers of {} are  {}% SENSING type'. format(option,round(100-NM[1]*100)))

if NM[2]>0.5:
    st.write('The readers of {} are  {}% FEELLING type'. format(option,round(NM[2]*100)))
else:
    st.write('The readers of {} are  {}% THINKING type'. format(option,round(100-NM[2]*100)))

if NM[3]>0.5:
    st.write('The readers of {} are  {}% PRECIEVING type'. format(option,round(NM[3]*100)))
else:
    st.write('The readers of {} are  {}% JUDGING type'. format(option,round(100-NM[3]*100)))
    
st.write(' Creator: Farhad Saffaraval, https://www.linkedin.com/in/farhad-saffaraval-ph-d-a802b829/ ')