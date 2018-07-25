import pandas as pd
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def loadDataset():
    dataset_bodies = pd.read_csv("../dataset/train_bodies.csv")
    dataset_stances = pd.read_csv("../dataset/train_stances.csv")
    
    bodies_dict = dict(zip(dataset_bodies['Body ID'], dataset_bodies['articleBody']))
    headline_body_pairs = dataset_stances.iloc[:, 0:2].values
    
    for index in range(0, len(headline_body_pairs)):
        # replacing body id with actual body
        headline_body_pairs[index,1] = bodies_dict.get(headline_body_pairs[index,1])
    
    # array that holds individual stances for each headline-body pair
    stances = dataset_stances.iloc[:, 2]
    
    return headline_body_pairs, stances
    
def filterStopwords(tokenized_doc, filter = "all"):
    if filter == "all":
        stops = stopwords.words("english")
        stops.extend(["\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",", "-", ".", "/", ":",
                      ";", "<", ">", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~", "=", "+", "!", "?", "\n", "’", "``", "''", "...", "'", "\"", "'"])
    elif filter == "punctonly":
        stops = []
        stops.extend(["\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",", "-", ".", "/", ":",
                      ";", "<", ">", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~", "\n", "’", "``", "''", "...", "'"])
    
    stops = set(stops)
    return [w for w in tokenized_doc if (not w in stops)]

def tokenizeDoc(doc, stopwords = "all"):
    if stopwords == "none":
        return word_tokenize(doc.lower())
    return filterStopwords(word_tokenize(doc.lower()), stopwords)

def processDataset(stopwords = "all"):
    headline_body_pairs, stances = loadDataset()
    
    for i in range(0, len(headline_body_pairs)):
        headline_body_pairs[i][0] = tokenizeDoc(headline_body_pairs[i][0], stopwords)
        headline_body_pairs[i][1] = tokenizeDoc(headline_body_pairs[i][1], stopwords)
       
    return headline_body_pairs, stances

def writeProcessedDatasetToFileSystem():
    headline_body_pairs, stances = processDataset("all")
    
    with open("../dataset/headline_body_pairs.csv", 'w', encoding='utf-16', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter = ',')
        filewriter.writerow(["Headline", "Body"])
        for i in range(0, len(headline_body_pairs)):
            headline = " ".join(headline_body_pairs[i][0])
            body = " ".join(headline_body_pairs[i][1])
            filewriter.writerow([headline, body]) 
    
    with open("../dataset/stances.csv", 'w', encoding='utf-16', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter = ' ')
        filewriter.writerow(["Stance"])
        for i in range(0, len(stances)):
            stance = "".join(stances[i])
            filewriter.writerow([stance])
    