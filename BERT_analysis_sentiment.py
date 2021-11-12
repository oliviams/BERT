import pandas as pd
import numpy as np
import glob
import re
import statistics

import nltk
nltk.download('punkt')
from nltk import sent_tokenize

import pip
pip.main(['install', 'openpyxl'])
# pip.main(['install', 'pysentimiento'])
from pysentimiento import SentimentAnalyzer
sentiment_analyzer = SentimentAnalyzer(lang="en")

def text_preprocessing(text):
    text = text.lower()
    text = re.sub("@\\w+", "", text)
    text = re.sub("https?://.+", "", text)
    text = re.sub("\\d+\\w*\\d*", "", text)
    text = re.sub("#\\w+", "", text)
    return(text)

def process(row):
    res = sentiment_analyzer.predict(row['Sentence'])
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

x = list(range(1, 6))
# x = list(range(1, 701))
score_list_session = [] #each session, list of average of NEU, NEG, POS scores per webpage
filepaths=[]
participants=[]

for i in x:
    #folder = "/Users/chriskelly/Dropbox/Chris_Information_Seeking_Studies/Web_Browsing_Study/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    folder = "/Users/olivia/Dropbox//Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    filepaths.append(folder)

for path in filepaths: # will it break when it reaches filepath that doesnt exist?
    filelist = [file for file in glob.glob(str(path) + "/*.txt")]
    score_list = []
    if len(filelist) == 0:
        pass
    else: # may be worth printing last bit of filepath / iteration it is on to match up the session
        for file in filelist:
            with open(file, encoding = 'latin1') as f:
                lines = f.readlines() # lines is a list of all the lines in a given file
                sent_split = [sent_tokenize(x) for x in lines]
                flat_list_sent = [item for sublist in sent_split for item in sublist]
                flat_list_sent = [tab.replace("\t", "") for tab in flat_list_sent]
                flat_list_sent = [text_preprocessing(x) for x in flat_list_sent]

                df_lines = pd.DataFrame(flat_list_sent, columns=['Sentence'])
                df_lines = df_lines.join(df_lines.apply(process, axis=1)) # creates a dataframe per file with file columns: sentence, overall sentimes, sentiment score (NEU, NEG, POS)
                df_lines = df_lines.append(df_lines[['NEU', 'NEG', 'POS']].mean(), ignore_index=True)
                score_list.append([df_lines.iloc[-1]['NEU'], df_lines.iloc[-1]['NEG'], df_lines.iloc[-1]['POS']])
    score_list_session.append(score_list)
    participants.append(path.split("/")[-1])
    # print(score_list_session)

# for each sentiment, give a list of lists --> average webpage scores for that sentiment within list of sessions
NEU_session = session_list(score_list_session, 'NEU') # [[x, x, x, x, x], [x, x, x], [x, x, x, x, x, x, x]]
NEG_session = session_list(score_list_session, 'NEG')
POS_session = session_list(score_list_session, 'POS')

# for each sentiment, average for that sentiment per session
NEU_session_average = session_average(NEU_session) # gave error for empty list - but shouldn't contain empty values?
NEG_session_average = session_average(NEG_session)
POS_session_average = session_average(POS_session)

NEU_session_std = session_std(NEU_session)
NEG_session_std = session_std(NEG_session)
POS_session_std = session_std(POS_session)

df_averages = pd.DataFrame(
    {'NEU average': NEU_session_average,
     'NEG average': NEG_session_average,
     'POS average': POS_session_average
    })
df_averages['Top Sentiment'] = df_averages.idxmax(axis=1)
df_averages['Top Sentiment'] = df_averages['Top Sentiment'].str.replace('average', '')
df_averages.index = participants


df_std = pd.DataFrame(
    {'NEU std': NEU_session_std,
     'NEG std': NEG_session_std,
     'POS std': POS_session_std
    })
df_std.index = participants

print(df_averages)
df_averages.to_excel("sentiment_averages.xlsx")
print(df_std)
df_std.to_excel("sentiment_stds.xlsx")





