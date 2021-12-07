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
    text = re.sub("https?://.+"import pandas as pd
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

# DAY 1

# x = list(range(1, 6))
x = list(range(1, 701))
filepaths=[]
participants=[]

session_list = []
similarity_matrices = []
similarity_means_day_1 = []

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
    session_list.append(doc_list)

for doc in range(len(session_list)):
    try:
        similarity_matrices.append(get_similarity(session_list[doc]))
    except:
        similarity_matrices.append(np.nan)

for matrix in similarity_matrices:
    try:
        similarity_means_day_1.append(matrix[np.triu_indices_from(matrix, 1)].mean())
    except:
        similarity_means_day_1.append(np.nan)


# DAY 2

# x = list(range(1, 6))
x = list(range(1, 701))
filepaths=[]
participants=[]

session_list = []
similarity_matrices = []
similarity_means_day_2 = []

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
    session_list.append(doc_list)

for doc in range(len(session_list)):
    try:
        similarity_matrices.append(get_similarity(session_list[doc]))
    except:
        similarity_matrices.append(np.nan)

for matrix in similarity_matrices:
    try:
        similarity_means_day_2.append(matrix[np.triu_indices_from(matrix, 1)].mean())
    except:
        similarity_means_day_2.append(np.nan)


# DAY 3

# x = list(range(1, 6))
x = list(range(1, 701))
filepaths=[]
participants=[]

session_list = []
similarity_matrices = []
similarity_means_day_3 = []

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
    session_list.append(doc_list)

for doc in range(len(session_list)):
    try:
        similarity_matrices.append(get_similarity(session_list[doc]))
    except:
        similarity_matrices.append(np.nan)

for matrix in similarity_matrices:
    try:
        similarity_means_day_3.append(matrix[np.triu_indices_from(matrix, 1)].mean())
    except:
        similarity_means_day_3.append(np.nan)


# DAY 4

# x = list(range(1, 6))
x = list(range(1, 701))
filepaths=[]
participants=[]

session_list = []
similarity_matrices = []
similarity_means_day_4 = []

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
    session_list.append(doc_list)

for doc in range(len(session_list)):
    try:
        similarity_matrices.append(get_similarity(session_list[doc]))
    except:
        similarity_matrices.append(np.nan)

for matrix in similarity_matrices:
    try:
        similarity_means_day_4.append(matrix[np.triu_indices_from(matrix, 1)].mean())
    except:
        similarity_means_day_4.append(np.nan)


# DAY 5

# x = list(range(1, 6))
x = list(range(1, 701))
filepaths=[]
participants=[]

session_list = []
similarity_matrices = []
similarity_means_day_5 = []

for i in x:
    #folder = "/Users/chriskelly/Dropbox/Chris_Information_Seeking_Studies/Web_Browsing_Study/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    folder = "C:/Users/o.macmillan-scott/Desktop/Main_Study_Combined/Separate_Files_Day_5/" + str(i)
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
    session_list.append(doc_list)

for doc in range(len(session_list)):
    try:
        similarity_matrices.append(get_similarity(session_list[doc]))
    except:
        similarity_matrices.append(np.nan)

for matrix in similarity_matrices:
    try:
        similarity_means_day_5.append(matrix[np.triu_indices_from(matrix, 1)].mean())
    except:
        similarity_means_day_5.append(np.nan)


compiled_df = pd.DataFrame({'DAY 1': similarity_means_day_1, 'DAY 2': similarity_means_day_2, 'DAY 3': similarity_means_day_3, 'DAY 4': similarity_means_day_4, 'DAY 5': similarity_means_day_5})
compiled_df.index += 1
print(compiled_df)

compiled_df.to_excel("document_similarity.xlsx")