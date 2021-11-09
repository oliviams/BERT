from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np
import glob
import re
from statistics import mode

# import nltk
# nltk.download('punkt')
# from nltk import sent_tokenize


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
valence_scores = []
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

# def text_preprocessing(text):
#     text = text.lower()
#     text = re.sub("@\\w+", "", text)
#     text = re.sub("https?://.+", "", text)
#     text = re.sub("\\d+\\w*\\d*", "", text)
#     text = re.sub("#\\w+", "", text)
#     # text = re.sub("\U000+", "", text) # emojis?
#     # text = re.sub("\N{+", "", text) # emojis?
#     return(text)

filepaths=[]

for i in x:
    #folder = "/Users/chriskelly/Dropbox/Chris_Information_Seeking_Studies/Web_Browsing_Study/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    folder = "/Users/olivia/Dropbox//Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    filepaths.append(folder)


def process(row):
    res = sentiment_analyzer.predict(row['Line'])
    return pd.Series({'Sentiment': res.output, **res.probas})  # if we don't need all three scores, could just have highest (as have a column for top scoring sentiment)

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
                #lines = text_preprocessing(lines)  # need to make sure this applies to list, not only string
                #lines_list = []

                df_lines = pd.DataFrame(lines, columns=['Line'])
                df_lines = df_lines.join(df_lines.apply(process, axis=1)) # creates a dataframe per file with file columns: line, overall sentimes, sentiment score (NEU, NEG, POS)
                df_lines = df_lines.append(df_lines[['NEU', 'NEG', 'POS']].mean(), ignore_index=True)
                top_sent.append(str(df_lines['Sentiment'].mode()[0]))
                top_sent_score.append(df_lines.iloc[-1][str(df_lines['Sentiment'].mode()[0])])
                file_scores.append(str(df_lines['Sentiment'].mode()[0]))
                file_scores.append(df_lines.iloc[-1][str(df_lines['Sentiment'].mode()[0])]) # shouldn't we be printing column name of highest average
                valence_scores.append(file_scores) # need to separate this into individuals, currently list of lists but not subdivided by participant
                score_list.append([df_lines.iloc[-1]['NEU'], df_lines.iloc[-1]['NEG'], df_lines.iloc[-1]['POS']])
                print(df_lines)
                # print(top_sent)
                # print(top_sent_score)
                # print(valence_scores)
    print(score_list)
    sent_list.append(top_sent)
    top_sent_session.append(mode(top_sent))
    score_list_session.append(score_list)
    print(top_sent_session)


average_scores_session = []  # list of list of NEU, NEG, POS scores per session (i.e. each item is a list of three scores representing the session)

for session in score_list_session:
    session_scores = []
    for i in range(0, 3):
        x=0
        for page in session:
            x=x+page[i]
        session_scores.append(x / len(session))
    average_scores_session.append(x/len(session_scores))

print(average_scores_session)






