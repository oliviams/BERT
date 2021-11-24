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

import nltk
nltk.download('punkt')
from nltk import sent_tokenize


MODEL = f"cardiffnlp/twitter-roberta-base-emotion"
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

def process(row):   # giving an error with some files, e.g. Day 1 1URL5.txt
    encoded_input = tokenizer(row['Sentence'], return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return pd.Series({'anger': scores[0], 'joy': scores[1], 'optimism': scores[2], 'sadness': scores[3]})

def session_list(score_list, sent):
    empty_list_session = []
    for session in score_list:
        empty = []
        for page in session:
            if sent == 'anger':
                empty.append(page[0])
            elif sent == 'joy':
                empty.append(page[1])
            elif sent == 'optimism':
                empty.append(page[2])
            elif sent == 'sadness':
                empty.append(page[3])
        empty_list_session.append(empty)
    return(empty_list_session)

def session_average(session_list):
    averages = []
    for session in session_list:
        try:
            averages.append(np.nanmean(session)) # should just be np.mean, np.nanmean as temporary fix
        except:
            averages.append(np.nan)
    return(averages)

def session_std(session_list):
    stds = []
    for session in session_list:
        try:
            stds.append(np.nanstd(session)) # should just be np.std, np.nanstd as temporary fix
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

for path in filepaths: # will it break when it reaches filepath that doesnt exist?
    filelist = [file for file in glob.glob(str(path) + "/*.txt")]
    score_list = []
    if len(filelist) == 0:
        pass
    else: # may be worth printing last bit of filepath / iteration it is on to match up the session
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
                    df_lines = df_lines.append(df_lines[['anger', 'joy', 'optimism', 'sadness']].mean(), ignore_index=True)
                    score_list.append([df_lines.iloc[-1]['anger'], df_lines.iloc[-1]['joy'], df_lines.iloc[-1]['optimism'], df_lines.iloc[-1]['sadness']])
            except:
                score_list.append([np.nan, np.nan, np.nan])
    score_list_session.append(score_list)
    participants.append(path.split("/")[-1])
    # print(score_list_session)

# for each sentiment, give a list of lists --> average webpage scores for that sentiment within list of sessions
anger_session = session_list(score_list_session, 'anger') # [[x, x, x, x, x], [x, x, x], [x, x, x, x, x, x, x]]
joy_session = session_list(score_list_session, 'joy')
optimism_session = session_list(score_list_session, 'optimism')
sadness_session = session_list(score_list_session, 'sadness')

# for each sentiment, average for that sentiment per session
anger_session_average = session_average(anger_session) # gave error for empty list - but shouldn't contain empty values?
joy_session_average = session_average(joy_session)
optimism_session_average = session_average(optimism_session)
sadness_session_average = session_average(sadness_session)

anger_session_std = session_std(anger_session_average)
joy_session_std = session_std(joy_session_average)
optimism_session_std = session_std(optimism_session_average)
sadness_session_std = session_std(sadness_session_average)

df_averages = pd.DataFrame(
    {'anger average': anger_session_average,
     'joy average': joy_session_average,
     'optimism average': optimism_session_average,
     'sadness average': sadness_session_average
     })
df_averages['Top Sentiment'] = df_averages.idxmax(axis=1)
df_averages['Top Sentiment'] = df_averages['Top Sentiment'].str.replace('average', '')
df_averages.index = participants


df_std = pd.DataFrame(
    {'anger average': anger_session_std,
     'joy average': joy_session_std,
     'optimism average': optimism_session_std,
     'sadness average': sadness_session_std
    })
df_std.index = participants

print(df_averages)
df_averages.to_excel("emotion_averages_roberta_day_1.xlsx")
print(df_std)
df_std.to_excel("emotion_stds_roberta_day_1.xlsx")


# DAY 2

# x = list(range(1, 6))
x = list(range(1, 701))
score_list_session = [] #each session, list of average of NEU, NEG, POS scores per webpage
filepaths=[]
participants=[]

for i in x:
    #folder = "/Users/chriskelly/Dropbox/Chris_Information_Seeking_Studies/Web_Browsing_Study/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    folder = "C:/Users/o.macmillan-scott/Desktop/Main_Study_Combined/Separate_Files_Day_2/" + str(i)
    filepaths.append(folder)

for path in filepaths: # will it break when it reaches filepath that doesnt exist?
    filelist = [file for file in glob.glob(str(path) + "/*.txt")]
    score_list = []
    if len(filelist) == 0:
        pass
    else: # may be worth printing last bit of filepath / iteration it is on to match up the session
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
                    df_lines = df_lines.append(df_lines[['anger', 'joy', 'optimism', 'sadness']].mean(), ignore_index=True)
                    score_list.append([df_lines.iloc[-1]['anger'], df_lines.iloc[-1]['joy'], df_lines.iloc[-1]['optimism'], df_lines.iloc[-1]['sadness']])
            except:
                score_list.append([np.nan, np.nan, np.nan])
    score_list_session.append(score_list)
    participants.append(path.split("/")[-1])
    # print(score_list_session)

# for each sentiment, give a list of lists --> average webpage scores for that sentiment within list of sessions
anger_session = session_list(score_list_session, 'anger') # [[x, x, x, x, x], [x, x, x], [x, x, x, x, x, x, x]]
joy_session = session_list(score_list_session, 'joy')
optimism_session = session_list(score_list_session, 'optimism')
sadness_session = session_list(score_list_session, 'sadness')

# for each sentiment, average for that sentiment per session
anger_session_average = session_average(anger_session) # gave error for empty list - but shouldn't contain empty values?
joy_session_average = session_average(joy_session)
optimism_session_average = session_average(optimism_session)
sadness_session_average = session_average(sadness_session)

anger_session_std = session_std(anger_session_average)
joy_session_std = session_std(joy_session_average)
optimism_session_std = session_std(optimism_session_average)
sadness_session_std = session_std(sadness_session_average)

df_averages = pd.DataFrame(
    {'anger average': anger_session_average,
     'joy average': joy_session_average,
     'optimism average': optimism_session_average,
     'sadness average': sadness_session_average
     })
df_averages['Top Sentiment'] = df_averages.idxmax(axis=1)
df_averages['Top Sentiment'] = df_averages['Top Sentiment'].str.replace('average', '')
df_averages.index = participants


df_std = pd.DataFrame(
    {'anger average': anger_session_std,
     'joy average': joy_session_std,
     'optimism average': optimism_session_std,
     'sadness average': sadness_session_std
    })
df_std.index = participants

print(df_averages)
df_averages.to_excel("emotion_averages_roberta_day_2.xlsx")
print(df_std)
df_std.to_excel("emotion_stds_roberta_day_2.xlsx")


# DAY 3

# x = list(range(1, 6))
x = list(range(1, 701))
score_list_session = [] #each session, list of average of NEU, NEG, POS scores per webpage
filepaths=[]
participants=[]

for i in x:
    #folder = "/Users/chriskelly/Dropbox/Chris_Information_Seeking_Studies/Web_Browsing_Study/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    folder = "C:/Users/o.macmillan-scott/Desktop/Main_Study_Combined/Separate_Files_Day_3/" + str(i)
    filepaths.append(folder)

for path in filepaths: # will it break when it reaches filepath that doesnt exist?
    filelist = [file for file in glob.glob(str(path) + "/*.txt")]
    score_list = []
    if len(filelist) == 0:
        pass
    else: # may be worth printing last bit of filepath / iteration it is on to match up the session
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
                    df_lines = df_lines.append(df_lines[['anger', 'joy', 'optimism', 'sadness']].mean(), ignore_index=True)
                    score_list.append([df_lines.iloc[-1]['anger'], df_lines.iloc[-1]['joy'], df_lines.iloc[-1]['optimism'], df_lines.iloc[-1]['sadness']])
            except:
                score_list.append([np.nan, np.nan, np.nan])
    score_list_session.append(score_list)
    participants.append(path.split("/")[-1])
    # print(score_list_session)

# for each sentiment, give a list of lists --> average webpage scores for that sentiment within list of sessions
anger_session = session_list(score_list_session, 'anger') # [[x, x, x, x, x], [x, x, x], [x, x, x, x, x, x, x]]
joy_session = session_list(score_list_session, 'joy')
optimism_session = session_list(score_list_session, 'optimism')
sadness_session = session_list(score_list_session, 'sadness')

# for each sentiment, average for that sentiment per session
anger_session_average = session_average(anger_session) # gave error for empty list - but shouldn't contain empty values?
joy_session_average = session_average(joy_session)
optimism_session_average = session_average(optimism_session)
sadness_session_average = session_average(sadness_session)

anger_session_std = session_std(anger_session_average)
joy_session_std = session_std(joy_session_average)
optimism_session_std = session_std(optimism_session_average)
sadness_session_std = session_std(sadness_session_average)

df_averages = pd.DataFrame(
    {'anger average': anger_session_average,
     'joy average': joy_session_average,
     'optimism average': optimism_session_average,
     'sadness average': sadness_session_average
     })
df_averages['Top Sentiment'] = df_averages.idxmax(axis=1)
df_averages['Top Sentiment'] = df_averages['Top Sentiment'].str.replace('average', '')
df_averages.index = participants


df_std = pd.DataFrame(
    {'anger average': anger_session_std,
     'joy average': joy_session_std,
     'optimism average': optimism_session_std,
     'sadness average': sadness_session_std
    })
df_std.index = participants

print(df_averages)
df_averages.to_excel("emotion_averages_roberta_day_3.xlsx")
print(df_std)
df_std.to_excel("emotion_stds_roberta_day_3.xlsx")


# DAY 4

# x = list(range(1, 6))
x = list(range(1, 701))
score_list_session = [] #each session, list of average of NEU, NEG, POS scores per webpage
filepaths=[]
participants=[]

for i in x:
    #folder = "/Users/chriskelly/Dropbox/Chris_Information_Seeking_Studies/Web_Browsing_Study/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    folder = "C:/Users/o.macmillan-scott/Desktop/Main_Study_Combined/Separate_Files_Day_4/" + str(i)
    filepaths.append(folder)

for path in filepaths: # will it break when it reaches filepath that doesnt exist?
    filelist = [file for file in glob.glob(str(path) + "/*.txt")]
    score_list = []
    if len(filelist) == 0:
        pass
    else: # may be worth printing last bit of filepath / iteration it is on to match up the session
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
                    df_lines = df_lines.append(df_lines[['anger', 'joy', 'optimism', 'sadness']].mean(), ignore_index=True)
                    score_list.append([df_lines.iloc[-1]['anger'], df_lines.iloc[-1]['joy'], df_lines.iloc[-1]['optimism'], df_lines.iloc[-1]['sadness']])
            except:
                score_list.append([np.nan, np.nan, np.nan])
    score_list_session.append(score_list)
    participants.append(path.split("/")[-1])
    # print(score_list_session)

# for each sentiment, give a list of lists --> average webpage scores for that sentiment within list of sessions
anger_session = session_list(score_list_session, 'anger') # [[x, x, x, x, x], [x, x, x], [x, x, x, x, x, x, x]]
joy_session = session_list(score_list_session, 'joy')
optimism_session = session_list(score_list_session, 'optimism')
sadness_session = session_list(score_list_session, 'sadness')

# for each sentiment, average for that sentiment per session
anger_session_average = session_average(anger_session) # gave error for empty list - but shouldn't contain empty values?
joy_session_average = session_average(joy_session)
optimism_session_average = session_average(optimism_session)
sadness_session_average = session_average(sadness_session)

anger_session_std = session_std(anger_session_average)
joy_session_std = session_std(joy_session_average)
optimism_session_std = session_std(optimism_session_average)
sadness_session_std = session_std(sadness_session_average)

df_averages = pd.DataFrame(
    {'anger average': anger_session_average,
     'joy average': joy_session_average,
     'optimism average': optimism_session_average,
     'sadness average': sadness_session_average
     })
df_averages['Top Sentiment'] = df_averages.idxmax(axis=1)
df_averages['Top Sentiment'] = df_averages['Top Sentiment'].str.replace('average', '')
df_averages.index = participants


df_std = pd.DataFrame(
    {'anger average': anger_session_std,
     'joy average': joy_session_std,
     'optimism average': optimism_session_std,
     'sadness average': sadness_session_std
    })
df_std.index = participants

print(df_averages)
df_averages.to_excel("emotion_averages_roberta_day_4.xlsx")
print(df_std)
df_std.to_excel("emotion_stds_roberta_day_4.xlsx")


# DAY 5

# x = list(range(1, 6))
x = list(range(1, 701))
score_list_session = [] #each session, list of average of NEU, NEG, POS scores per webpage
filepaths=[]
participants=[]

for i in x:
    #folder = "/Users/chriskelly/Dropbox/Chris_Information_Seeking_Studies/Web_Browsing_Study/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    folder = "C:/Users/o.macmillan-scott/Desktop/Main_Study_Combined/Separate_Files_Day_5/" + str(i)
    filepaths.append(folder)

for path in filepaths: # will it break when it reaches filepath that doesnt exist?
    filelist = [file for file in glob.glob(str(path) + "/*.txt")]
    score_list = []
    if len(filelist) == 0:
        pass
    else: # may be worth printing last bit of filepath / iteration it is on to match up the session
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
                    df_lines = df_lines.append(df_lines[['anger', 'joy', 'optimism', 'sadness']].mean(), ignore_index=True)
                    score_list.append([df_lines.iloc[-1]['anger'], df_lines.iloc[-1]['joy'], df_lines.iloc[-1]['optimism'], df_lines.iloc[-1]['sadness']])
            except:
                score_list.append([np.nan, np.nan, np.nan])
    score_list_session.append(score_list)
    participants.append(path.split("/")[-1])
    # print(score_list_session)

# for each sentiment, give a list of lists --> average webpage scores for that sentiment within list of sessions
anger_session = session_list(score_list_session, 'anger') # [[x, x, x, x, x], [x, x, x], [x, x, x, x, x, x, x]]
joy_session = session_list(score_list_session, 'joy')
optimism_session = session_list(score_list_session, 'optimism')
sadness_session = session_list(score_list_session, 'sadness')

# for each sentiment, average for that sentiment per session
anger_session_average = session_average(anger_session) # gave error for empty list - but shouldn't contain empty values?
joy_session_average = session_average(joy_session)
optimism_session_average = session_average(optimism_session)
sadness_session_average = session_average(sadness_session)

anger_session_std = session_std(anger_session_average)
joy_session_std = session_std(joy_session_average)
optimism_session_std = session_std(optimism_session_average)
sadness_session_std = session_std(sadness_session_average)

df_averages = pd.DataFrame(
    {'anger average': anger_session_average,
     'joy average': joy_session_average,
     'optimism average': optimism_session_average,
     'sadness average': sadness_session_average
     })
df_averages['Top Sentiment'] = df_averages.idxmax(axis=1)
df_averages['Top Sentiment'] = df_averages['Top Sentiment'].str.replace('average', '')
df_averages.index = participants


df_std = pd.DataFrame(
    {'anger average': anger_session_std,
     'joy average': joy_session_std,
     'optimism average': optimism_session_std,
     'sadness average': sadness_session_std
    })
df_std.index = participants

print(df_averages)
df_averages.to_excel("emotion_averages_roberta_day_5.xlsx")
print(df_std)
df_std.to_excel("emotion_stds_roberta_day_5.xlsx")