"""
Authors: Lamiaa Dakir
Date: 12/17/2019
Description: script to parse an email in the form of the features from the spamdata.
"""

import os
import numpy as np

def load_email(filename):
    """
    Loads the email and splits it into words.
    """
    f = open(filename, 'r',  encoding="ISO-8859-1")
    data = []
    for x in f:
        words = x.split()
        for w in words:
            data.append(w)
    return data

def parse_email(email, label):

    #loading parsed email
    data = load_email(email)

    #words taken from database documentation
    words = ['make', 'address', 'all', '3d', 'our', 'over' , 'remove', 'internet','order', 'mail', 'receive', 'will', 'people', 'report',\
    'addresses','free', 'business', 'email', 'you', 'credit', 'your', 'font', '000', 'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet',\
    '857', 'data', '415', '85', 'technology', '1999', 'parts', 'pm', 'direct', 'cs', 'meeting', 'original', 'project', 're', 'edu', 'table', 'conference', ';', '(',\
    '[', '!','$', '#']

    #finding the frequency of the words abd characters
    features = []
    freq ={}
    for x in data:
        if x in words:
            if x in freq:
                freq[x] +=1
            else:
                freq[x] = 1
    for w in words:
        if w in freq:
            features.append(freq[w])
        else:
            features.append(0)


    #finding capital letters in the email
    cap_words_len = []
    cap_num = 0
    for word in data:
        cap_in_word = 0
        for char in word:
            if char.isupper():
                cap_in_word += 1
                cap_num +=1
        cap_words_len.append(cap_in_word)

    max_cap_len =  max(cap_words_len)
    cap_average =  sum(cap_words_len)/len(cap_words_len)

    #adding capital letters statistic to the features
    features.append(cap_average)
    features.append(max_cap_len)
    features.append(cap_num)

    # Creating array of one example
    examples = []
    examples.append(features)


    #0 is the email is not spam and 1 if it is a spam
    labels = []
    labels.append(label)

    return examples, labels
