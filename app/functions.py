import numpy as np
import pandas as pd
import re
from collections import Counter

def df_to_lower(df):
    """
    Args:
        df (DataFrame): Data to be converted to lowercase
    
    Returns:
        df (DataFrame): Converted dataframe in lowercase
    """
    df['subject'] = df['subject'].str.lower()
    df['email'] = df['email'].str.lower()
    return df

def words_in_texts(words, texts):
    """
    Args:
        words (list): Words to find.
        texts (Series): Strings to search in.

    Returns:
        indicator_array (np.ndarray): A 2D NumPy array
        of binary values with shape (n, d) where n is
        the number of texts, and d is the number of 
        words.
    """
    indicator_array = np.array([texts.str.contains(word) for word in words]).T.astype('int')

    return indicator_array

def most_occurring_word(email):
    """
    Args:
        email (Series): Series of email body text data
    
    Returns:
        most_common (string): Most occurring word in an email body
    """
    words = re.findall(r'\w+', email.lower())
    word_counter = Counter(words)

    most_common = max(word_counter, key=word_counter.get)

    return most_common

def transformations(df):
    df['html_tags'] = df['email'].str.count('<.*?\>')
    df['body_characters'] = df['email'].str.len()
    df['body_length'] = df['email'].str.split().str.len()
    df['subj_length'] = df['subject'].str.split().str.len()
    df['exclamations'] = df['email'].str.count('!')
    df['is_reply'] = df['subject'].str.contains('re:').astype('int')
    df['has_ip'] = df['email'].str.contains('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}').astype('int')
    return df