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

x = list(range(1, 6))
# x = list(range(1, 701))
filepaths=[]
participants=[]

doc_list = []
similarity_matrices = []

for i in x:
    #folder = "/Users/chriskelly/Dropbox/Chris_Information_Seeking_Studies/Web_Browsing_Study/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    folder = "C:/Users/o.macmillan-scott/Desktop/Main_Study_Combined/Separate_Files_Day_1/" + str(i)
    filepaths.append(folder)

for path in filepaths:
    filelist = [file for file in glob.glob(str(path) + "/*.txt")]
    full_doc = []
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
                full_doc.append(all_lines)
    doc_list.append(full_doc)

for doc in range(len(doc_list)):
    try:
        similarity_matrices.append(get_similarity(doc_list[doc]))
    except:
        similarity_matrices.append(np.nan)

print(similarity_matrices)