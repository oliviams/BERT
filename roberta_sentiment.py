from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request

import pandas as pd
import glob
import re
import statistics

import nltk
nltk.download('punkt')
from nltk import sent_tokenize

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)

def text_preprocessing(text):
    text = text.lower()
    text = re.sub("@\\w+", "", text)
    text = re.sub("https?://.+", "", text)
    text = re.sub("\\d+\\w*\\d*", "", text)
    text = re.sub("#\\w+", "", text)
    return(text)


def process(row):
    encoded_input = tokenizer(row['Sentence'], return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return pd.Series({'NEG': scores[0], 'NEU': scores[1], 'POS': scores[2]})

def session_list(score_list, sent):
    empty_list_session = []
    for session in score_list:
        empty = []
        for page in session:
            if sent == 'NEG':
                empty.append(page[0])
            elif sent == 'NEU':
                empty.append(page[1])
            elif sent == 'POS':
                empty.append(page[2])
        empty_list_session.append(empty)
    return(empty_list_session)

def session_average(session_list):
    averages = []
    for session in session_list:
        try:
            averages.append(statistics.mean(session))
        except:
            averages.append(np.nan)
    return(averages)

def session_std(session_list):
    stds = []
    for session in session_list:
        try:
            stds.append(statistics.stdev(session))
        except:
            stds.append(np.nan)
    return(stds)

# x = list(range(1, 6))
x = list(range(1, 701))
score_list_session = [] #each session, list of average of NEU, NEG, POS scores per webpage
filepaths=[]
participants=[]

for i in x:
    #folder = "/Users/chriskelly/Dropbox/Chris_Information_Seeking_Studies/Web_Browsing_Study/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    folder = "C:/Users/o.macmillan-scott/Desktop/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    filepaths.append(folder)

for path in filepaths:
    filelist = [file for file in glob.glob(str(path) + "/*.txt")]
    score_list = []
    if len(filelist) == 0:
        pass
    else:
        for file in filelist:
            try:
                with open(file, encoding = 'latin1') as f:
                    lines = f.readlines() # lines is a list of all the lines in a given file
                    sent_split = [sent_tokenize(x) for x in lines]
                    flat_list_sent = [item for sublist in sent_split for item in sublist]
                    flat_list_sent = [tab.replace("\t", "") for tab in flat_list_sent]
                    flat_list_sent = [text_preprocessing(x) for x in flat_list_sent]

                    df_lines = pd.DataFrame(flat_list_sent, columns=['Sentence'])
                    df_lines = df_lines.join(df_lines.apply(process, axis=1)) # creates a dataframe per file with file columns: sentence, sentiment score (NEU, NEG, POS)
                    df_lines = df_lines.append(df_lines[['NEG', 'NEU', 'POS']].mean(), ignore_index=True)
                    score_list.append([df_lines.iloc[-1]['NEG'], df_lines.iloc[-1]['NEU'], df_lines.iloc[-1]['POS']])
            except:
                score_list.append([np.nan, np.nan, np.nan])
    score_list_session.append(score_list)
    participants.append(path.split("/")[-1])
    # print(score_list_session)

# for each sentiment, give a list of lists --> average webpage scores for that sentiment within list of sessions
NEG_session = session_list(score_list_session, 'NEG')
NEU_session = session_list(score_list_session, 'NEU')
POS_session = session_list(score_list_session, 'POS')

# for each sentiment, average for that sentiment per session
NEG_session_average = session_average(NEG_session) 
NEU_session_average = session_average(NEU_session)
POS_session_average = session_average(POS_session)

NEG_session_std = session_std(NEG_session)
NEU_session_std = session_std(NEU_session)
POS_session_std = session_std(POS_session)

df_averages = pd.DataFrame(
    {'NEG average': NEG_session_average,
     'NEU average': NEU_session_average,
     'POS average': POS_session_average
    })
df_averages['Top Sentiment'] = df_averages.idxmax(axis=1)
df_averages['Top Sentiment'] = df_averages['Top Sentiment'].str.replace('average', '')
df_averages.index = participants


df_std = pd.DataFrame(
    {'NEG std': NEG_session_std,
     'NEU std': NEU_session_std,
     'POS std': POS_session_std
    })
df_std.index = participants

print(df_averages)
df_averages.to_excel("sentiment_averages.xlsx")
print(df_std)
df_std.to_excel("sentiment_stds.xlsx")