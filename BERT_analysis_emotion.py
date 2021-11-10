from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np
import glob
import re
from statistics import mode
import statistics

import nltk
nltk.download('punkt')
from nltk import sent_tokenize


import pip
pip.main(['install', 'pysentimiento'])
from pysentimiento import EmotionAnalyzer
emotion_analyzer = EmotionAnalyzer(lang="en")


x = list(range(1, 3))
#x = list(range(1, 701))
valence_scores = [] #  list of lists of top sentiment and the average for that sentiment
sent_list = [] # list with top sentiment per webpage within list of sessions
top_sent_session = [] # top sentiment per session
score_list_session = [] #each session, list of average of NEU, NEG, POS scores per webpage

def text_preprocessing(text):
    text = text.lower()
    text = re.sub("@\\w+", "", text)
    text = re.sub("https?://.+", "", text)
    text = re.sub("\\d+\\w*\\d*", "", text)
    text = re.sub("#\\w+", "", text)
    # text = re.sub("\U000+", "", text) # emojis?
    # text = re.sub("\N{+", "", text) # emojis?
    return(text)

filepaths=[]

for i in x:
    #folder = "/Users/chriskelly/Dropbox/Chris_Information_Seeking_Studies/Web_Browsing_Study/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    folder = "/Users/olivia/Dropbox//Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    filepaths.append(folder)

def process(row):
    res = emotion_analyzer.predict(row['Line'])
    return pd.Series({'Sentiment': res.output, **res.probas})

def session_list(score_list, sent):
    empty_list_session = []
    for session in score_list:
        empty = []
        for page in session:
            if sent == 'joy':
                empty.append(page[0])  # check order of emotions
            elif sent == 'anger':
                empty.append(page[1])
            elif sent == 'sadness':
                empty.append(page[2])
            elif sent == 'fear':
                empty.append(page[3])
            elif sent == 'disgust':
                empty.append(page[4])
            elif sent == 'surprise':
                empty.append(page[5])
            elif sent == 'others':
                empty.append(page[6])
        empty_list_session.append(empty)
    return(empty_list_session)

def session_average(session_list, sent):
    averages = []
    for session in session_list:
        averages.append(statistics.mean(session))
    return(averages)

def session_std(session_list, sent):
    stds = []
    for session in session_list:
        stds.append(statistics.stdev(session))
    return(stds)

for path in filepaths:
    filelist = [file for file in glob.glob(str(path) + "/*.txt")]
    top_sent = []
    top_sent_score = []
    score_list = []
    if len(filelist) == 0:
        pass
    else:
        for file in filelist:
            file_scores = []
            with open(file, encoding = 'latin1') as f:
                lines = f.readlines() # lines is a list of all the lines in a given file
                sent_split = [sent_tokenize(x) for x in lines]
                flat_list_sent = [item for sublist in sent_split for item in sublist]
                flat_list_sent = [tab.replace("\t", "") for tab in flat_list_sent]
                flat_list_sent = [text_preprocessing(x) for x in flat_list_sent]

                df_lines = pd.DataFrame(flat_list_sent, columns=['Line'])
                df_lines = df_lines.join(df_lines.apply(process, axis=1)) # creates a dataframe per file with file columns: sentence, overall sentimes, sentiment score (NEU, NEG, POS)
                df_lines = df_lines.append(df_lines[['joy', 'anger', 'sadness', 'fear', 'disgust', 'surprise', 'others']].mean(), ignore_index=True)
                top_sent.append(str(df_lines.iloc[-1].astype(float).idxmax()))
                top_sent_score.append(df_lines.iloc[-1][str(df_lines.iloc[-1].astype(float).idxmax())])
                file_scores.append(str(df_lines.iloc[-1].astype(float).idxmax()))
                file_scores.append(df_lines.iloc[-1][str(df_lines.iloc[-1].astype(float).idxmax())])
                valence_scores.append(file_scores)
                score_list.append([df_lines.iloc[-1]['joy'], df_lines.iloc[-1]['anger'], df_lines.iloc[-1]['sadness'], df_lines.iloc[-1]['fear'], df_lines.iloc[-1]['disgust'], df_lines.iloc[-1]['surprise'], df_lines.iloc[-1]['others']])
                print(df_lines)
    sent_list.append(top_sent)
    # top_sent_session.append(mode(top_sent))
    score_list_session.append(score_list)
    print(top_sent_session)

joy_session = session_list(score_list_session, 'joy')
anger_session = session_list(score_list_session, 'anger')
sadness_session = session_list(score_list_session, 'sadness')
fear_session = session_list(score_list_session, 'fear')
disgust_session = session_list(score_list_session, 'disgust')
surprise_session = session_list(score_list_session, 'surprise')
others_session = session_list(score_list_session, 'others')

joy_session_average = session_average(joy_session, 'joy')
anger_session_average = session_average(anger_session, 'anger')
sadness_session_average = session_average(sadness_session, 'sadness')
fear_session_average = session_average(fear_session, 'fear')
disgust_session_average = session_average(disgust_session, 'disgust')
surprise_session_average = session_average(surprise_session, 'surprise')
others_session_average = session_average(others_session, 'others')

joy_session_std = session_std(joy_session, 'joy')
anger_session_std = session_std(anger_session, 'anger')
sadness_session_std = session_std(sadness_session, 'sadness')
fear_session_std = session_std(fear_session, 'fear')
disgust_session_std = session_std(disgust_session, 'disgust')
surprise_session_std = session_std(surprise_session, 'surprise')
others_session_std = session_std(others_session, 'others')

df_averages = pd.DataFrame(
    {'joy average': joy_session_average,
     'anger average': anger_session_average,
     'sadness average': sadness_session_average,
     'fear average': fear_session_average,
     'disgust average': disgust_session_average,
     'surprise average': surprise_session_average,
     'others average': others_session_average
    })

df_std = pd.DataFrame(
    {'joy std': joy_session_std,
     'anger std': anger_session_std,
     'sadness std': sadness_session_std,
     'fear std': fear_session_std,
     'disgust std': disgust_session_std,
     'surprise std': surprise_session_std,
     'others std': others_session_std
    })

print(df_averages)
print(df_std)


