import pandas as pd
import numpy as np
import glob
import re

import nltk
nltk.download('punkt')
from nltk import sent_tokenize

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def text_preprocessing(text):
    text = text.lower()
    text = re.sub("@\\w+", "", text)
    text = re.sub("https?://.+", "", text)
    text = re.sub("\\d+\\w*\\d*", "", text)
    text = re.sub("#\\w+", "", text)
    return(text)

def get_similarity(document_list):
    embeddings = model.encode(document_list)
    return cosine_similarity(embeddings)

x = list(range(1, 11))
# x = list(range(1, 701))


# DAY 1
filepaths=[]

session_list_day_1 = []
similarity_matrices = []

for i in x:
    #folder = "/Users/chriskelly/Dropbox/Chris_Information_Seeking_Studies/Web_Browsing_Study/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    folder = "C:/Users/o.macmillan-scott/Desktop/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    filepaths.append(folder)

for path in filepaths:
    filelist = [file for file in glob.glob(str(path) + "/*.txt")]
    doc_list = []
    if len(filelist) == 0:
        pass
    else:
        for file in filelist:
            with open(file, encoding = 'latin1') as f:
                lines = f.readlines() # lines is a list of all the lines in a given file
                sent_split = [sent_tokenize(x) for x in lines]
                flat_list_sent = [item for sublist in sent_split for item in sublist]
                flat_list_sent = [tab.replace("\t", "") for tab in flat_list_sent]
                flat_list_sent = [text_preprocessing(x) for x in flat_list_sent]
                all_lines = ' '.join(flat_list_sent)
                doc_list.append(all_lines)
    session = ''.join(doc_list)
    session_list_day_1.append(session)



# DAY 2

filepaths=[]

session_list_day_2 = []
similarity_matrices = []

for i in x:
    #folder = "/Users/chriskelly/Dropbox/Chris_Information_Seeking_Studies/Web_Browsing_Study/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    folder = "C:/Users/o.macmillan-scott/Desktop/Main_Study_Combined/Separate_Files_Day_2/" + str(i)
    filepaths.append(folder)

for path in filepaths:
    filelist = [file for file in glob.glob(str(path) + "/*.txt")]
    doc_list = []
    if len(filelist) == 0:
        pass
    else:
        for file in filelist:
            with open(file, encoding = 'latin1') as f:
                lines = f.readlines() # lines is a list of all the lines in a given file
                sent_split = [sent_tokenize(x) for x in lines]
                flat_list_sent = [item for sublist in sent_split for item in sublist]
                flat_list_sent = [tab.replace("\t", "") for tab in flat_list_sent]
                flat_list_sent = [text_preprocessing(x) for x in flat_list_sent]
                all_lines = ' '.join(flat_list_sent)
                doc_list.append(all_lines)
    session = ''.join(doc_list)
    session_list_day_2.append(session)



# DAY 3

filepaths=[]

session_list_day_3 = []
similarity_matrices = []

for i in x:
    #folder = "/Users/chriskelly/Dropbox/Chris_Information_Seeking_Studies/Web_Browsing_Study/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    folder = "C:/Users/o.macmillan-scott/Desktop/Main_Study_Combined/Separate_Files_Day_3/" + str(i)
    filepaths.append(folder)

for path in filepaths:
    filelist = [file for file in glob.glob(str(path) + "/*.txt")]
    doc_list = []
    if len(filelist) == 0:
        pass
    else:
        for file in filelist:
            with open(file, encoding = 'latin1') as f:
                lines = f.readlines() # lines is a list of all the lines in a given file
                sent_split = [sent_tokenize(x) for x in lines]
                flat_list_sent = [item for sublist in sent_split for item in sublist]
                flat_list_sent = [tab.replace("\t", "") for tab in flat_list_sent]
                flat_list_sent = [text_preprocessing(x) for x in flat_list_sent]
                all_lines = ' '.join(flat_list_sent)
                doc_list.append(all_lines)
    session = ''.join(doc_list)
    session_list_day_3.append(session)


# DAY 4

filepaths=[]

session_list_day_4 = []
similarity_matrices = []

for i in x:
    #folder = "/Users/chriskelly/Dropbox/Chris_Information_Seeking_Studies/Web_Browsing_Study/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    folder = "C:/Users/o.macmillan-scott/Desktop/Main_Study_Combined/Separate_Files_Day_4/" + str(i)
    filepaths.append(folder)

for path in filepaths:
    filelist = [file for file in glob.glob(str(path) + "/*.txt")]
    doc_list = []
    if len(filelist) == 0:
        pass
    else:
        for file in filelist:
            with open(file, encoding = 'latin1') as f:
                lines = f.readlines() # lines is a list of all the lines in a given file
                sent_split = [sent_tokenize(x) for x in lines]
                flat_list_sent = [item for sublist in sent_split for item in sublist]
                flat_list_sent = [tab.replace("\t", "") for tab in flat_list_sent]
                flat_list_sent = [text_preprocessing(x) for x in flat_list_sent]
                all_lines = ' '.join(flat_list_sent)
                doc_list.append(all_lines)
    session = ''.join(doc_list)
    session_list_day_4.append(session)


# DAY 5

filepaths = []

session_list_day_5 = []
similarity_matrices = []

for i in x:
    # folder = "/Users/chriskelly/Dropbox/Chris_Information_Seeking_Studies/Web_Browsing_Study/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    folder = "C:/Users/o.macmillan-scott/Desktop/Main_Study_Combined/Separate_Files_Day_5/" + str(i)
    filepaths.append(folder)

for path in filepaths:
    filelist = [file for file in glob.glob(str(path) + "/*.txt")]
    doc_list = []
    if len(filelist) == 0:
        pass
    else:
        for file in filelist:
            with open(file, encoding='latin1') as f:
                lines = f.readlines()  # lines is a list of all the lines in a given file
                sent_split = [sent_tokenize(x) for x in lines]
                flat_list_sent = [item for sublist in sent_split for item in sublist]
                flat_list_sent = [tab.replace("\t", "") for tab in flat_list_sent]
                flat_list_sent = [text_preprocessing(x) for x in flat_list_sent]
                all_lines = ' '.join(flat_list_sent)
                doc_list.append(all_lines)
    session = ''.join(doc_list)
    session_list_day_5.append(session)


for (day1, day2, day3, day4, day5) in zip(session_list_day_1, session_list_day_2, session_list_day_3, session_list_day_4, session_list_day_5):
    participant_days = []
    if day1 != '':
        participant_days.append(day1)
    if day2 != '':
        participant_days.append(day2)
    if day3 != '':
        participant_days.append(day3)
    if day4 != '':
        participant_days.append(day4)
    if day5 != '':
        participant_days.append(day5)
    try:
        similarity_matrices.append(get_similarity(participant_days))
    except:
        similarity_matrices.append(np.nan)

print(similarity_matrices)

similarity_means = []

for matrix in similarity_matrices:
    try:
        similarity_means.append(matrix[np.triu_indices_from(matrix, 1)].mean())
    except:
        similarity_means.append(np.nan)

print(similarity_means)

final_df = pd.DataFrame({'Similarity score': similarity_means})
final_df.index += 1
print(final_df)

final_df.to_excel("participant_similarity.xlsx")