# -*- coding: utf-8 -*-
"""
Created on Tue May 30 20:03:39 2017

@author: Sunny Lam
"""

from nltk import word_tokenize, FreqDist, bigrams, everygrams, trigrams
from nltk.corpus import stopwords

def token_count(series):
    return series.apply(lambda x: len(word_tokenize(str(x))))
    
def character_count(series):
    return series.apply(lambda x: str(x).replace(" ", "")).apply(lambda x: len(x))

def mean_characters_per_word(series):
    return character_count(series).divide(word_count(series))

def unique_vocabulary_count(series):
    return series.apply(str).apply(lambda x: len(set([word for word in word_tokenize(x)])))

def lexical_diversity(series):
    return series.apply(str).apply(lambda x: len(set([word for word in word_tokenize(x) if word not in stopwords.words("english")])) / len([word for word in word_tokenize(x) if word not in stopwords.words("english")]))

def word_is_present(series, word):
    return series.apply(str).apply(str.lower).apply(lambda x: 1 if word in x else 0)

def item_count(series):
    return series.apply(str).apply(lambda x: len(x.replace("/", ",").split(",")))

def add_word_presence_features(data,text_feature, word_list):
    
    for w in word_list: 
        key = text_feature + "/" + w
        dictionary = { key : word_is_present(data[text_feature],w)}
        data[key] = word_is_present(data[text_feature],w)
        
    return data

def find_most_common(series):
    series = series.apply(str).apply(str.lower).tolist()
    raw_text = " ".join(series)
    tokens = word_tokenize(raw_text)
    tokens = [token for token in tokens if token not in stopwords.words("english") + [",",".","&"]]
    grams = everygrams(tokens, max_len=1)
    frequencies = FreqDist(grams)
    return frequencies.most_common(60)