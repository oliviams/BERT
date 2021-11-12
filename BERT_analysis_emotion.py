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
pip.main(['install', 'pysentimiento'])
from pysentimiento import EmotionAnalyzer
emotion_analyzer = EmotionAnalyzer(lang="en")

def text_preprocessing(text):
    text = text.lower()
    text = re.sub("@\\w+", "", text)
    text = re.sub("https?://.+", "", text)
    text = re.sub("\\d+\\w*\\d*", "", text)
    text = re.sub("#\\w+", "", text)
    return(text)

def process(row):
    res = emotion_analyzer.predict(row['Sentence'])
    return pd.Series({'Emotion': res.output, **res.probas})

def session_list(score_list, emotion):
    empty_list_session = []
    for session in score_list:
        empty = []
        for page in session:
            if emotion == 'joy':
                empty.append(page[0])  # check order of emotions
            elif emotion == 'anger':
                empty.append(page[1])
            elif emotion == 'sadness':
                empty.append(page[2])
            elif emotion == 'fear':
                empty.append(page[3])
            elif emotion == 'disgust':
                empty.append(page[4])
            elif emotion == 'surprise':
                empty.append(page[5])
            elif emotion == 'others':
                empty.append(page[6])
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

for path in filepaths:
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
                df_lines = df_lines.append(df_lines[['joy', 'anger', 'sadness', 'fear', 'disgust', 'surprise', 'others']].mean(), ignore_index=True)
                score_list.append([df_lines.iloc[-1]['joy'], df_lines.iloc[-1]['anger'], df_lines.iloc[-1]['sadness'], df_lines.iloc[-1]['fear'], df_lines.iloc[-1]['disgust'], df_lines.iloc[-1]['surprise'], df_lines.iloc[-1]['others']])
    score_list_session.append(score_list)
    participants.append(path.split("/")[-1])
    # print(score_list_session)

# for each sentiment, give a list of lists --> average webpage scores for that sentiment within list of sessions
joy_session = session_list(score_list_session, 'joy')
anger_session = session_list(score_list_session, 'anger')
sadness_session = session_list(score_list_session, 'sadness')
fear_session = session_list(score_list_session, 'fear')
disgust_session = session_list(score_list_session, 'disgust')
surprise_session = session_list(score_list_session, 'surprise')
others_session = session_list(score_list_session, 'others')

# for each sentiment, average for that sentiment per session
joy_session_average = session_average(joy_session)
anger_session_average = session_average(anger_session)
sadness_session_average = session_average(sadness_session)
fear_session_average = session_average(fear_session)
disgust_session_average = session_average(disgust_session)
surprise_session_average = session_average(surprise_session)
others_session_average = session_average(others_session)

joy_session_std = session_std(joy_session)
anger_session_std = session_std(anger_session)
sadness_session_std = session_std(sadness_session)
fear_session_std = session_std(fear_session)
disgust_session_std = session_std(disgust_session)
surprise_session_std = session_std(surprise_session)
others_session_std = session_std(others_session)

df_averages = pd.DataFrame(
    {'joy average': joy_session_average,
     'anger average': anger_session_average,
     'sadness average': sadness_session_average,
     'fear average': fear_session_average,
     'disgust average': disgust_session_average,
     'surprise average': surprise_session_average,
     'others average': others_session_average
    })
df_averages['Top Emotion'] = df_averages.idxmax(axis=1)
df_averages['Top Emotion'] = df_averages['Top Emotion'].str.replace('average', '')
df_averages.index = participants

df_std = pd.DataFrame(
    {'joy std': joy_session_std,
     'anger std': anger_session_std,
     'sadness std': sadness_session_std,
     'fear std': fear_session_std,
     'disgust std': disgust_session_std,
     'surprise std': surprise_session_std,
     'others std': others_session_std
    })
df_std.index = participants

print(df_averages)
df_averages.to_excel("emotion_averages.xlsx")
print(df_std)
df_std.to_excel("emotion_stds.xlsx")






