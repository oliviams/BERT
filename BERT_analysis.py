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
from pysentimiento import SentimentAnalyzer
sentiment_analyzer = SentimentAnalyzer(lang="en")


x = list(range(1, 3))
#x = list(range(1, 701))
# day1 = []
# day2 = []
# day3 = []
# day4 = []
# day5 = []
valence_scores = [] #  list of lists of top sentiment and the average for that sentiment
sent_list = [] # list with top sentiment per webpage within list of sessions
top_sent_session = [] # top sentiment per session
score_list_session = [] #each session, list of average of NEU, NEG, POS scores per webpage

# tokenizer = AutoTokenizer.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis')
# model = AutoModelForSequenceClassification.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis')
#
# def valence_score(text):
#     tokens = tokenizer.encode(text, return_tensors='pt')
#     result = model(tokens)
#     return int(torch.argmax(result.logits))+1

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
    res = sentiment_analyzer.predict(row['Line'])
    return pd.Series({'Sentiment': res.output, **res.probas})

def session_list(score_list, sent):
    empty_list_session = []
    for session in score_list:
        empty = []
        for page in session:
            if sent == 'NEU':
                empty.append(page[0])
            elif sent == 'NEG':
                empty.append(page[1])
            elif sent == 'POS':
                empty.append(page[2])
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
                df_lines = df_lines.append(df_lines[['NEU', 'NEG', 'POS']].mean(), ignore_index=True)
                top_sent.append(str(df_lines.iloc[-1].astype(float).idxmax()))
                top_sent_score.append(df_lines.iloc[-1][str(df_lines.iloc[-1].astype(float).idxmax())])
                file_scores.append(str(df_lines.iloc[-1].astype(float).idxmax()))
                file_scores.append(df_lines.iloc[-1][str(df_lines.iloc[-1].astype(float).idxmax())])
                valence_scores.append(file_scores)
                score_list.append([df_lines.iloc[-1]['NEU'], df_lines.iloc[-1]['NEG'], df_lines.iloc[-1]['POS']])
    sent_list.append(top_sent)
    # top_sent_session.append(mode(top_sent))
    score_list_session.append(score_list)
    print(top_sent_session)

NEU_session = session_list(score_list_session, 'NEU')
NEG_session = session_list(score_list_session, 'NEG')
POS_session = session_list(score_list_session, 'POS')

NEU_session_average = session_average(NEU_session, 'NEU')
NEG_session_average = session_average(NEG_session, 'NEG')
POS_session_average = session_average(POS_session, 'POS')

NEU_session_std = session_std(NEU_session, 'NEU')
NEG_session_std = session_std(NEG_session, 'NEG')
POS_session_std = session_std(POS_session, 'POS')

# Need to do the same for emotion






