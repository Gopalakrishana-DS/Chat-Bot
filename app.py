#!/usr/bin/env python
# coding: utf-8

# In[144]:

import streamlit as st
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import json
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')


data = pd.read_csv('dialogs.txt' , sep='\t' , names=['Question' , 'Answer'])

question_list = data['Question'].tolist()
answer_list = data['Answer'].tolist()

def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text = re.sub(r'[^\w\s]', '' , text)
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatize_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatize_tokens]
    return ' '.join(stemmed_tokens)

def preprocess_with_stopwords(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text = re.sub(r'[^\w\s]', '' , text)
    tokens = nltk.word_tokenize(text.lower())
    lemmatize_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatize_tokens]
    return ' '.join(stemmed_tokens)


corpus = question_list + answer_list
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform([preprocess(text) for text in corpus])


def get_response(text):
    processed_text = preprocess_with_stopwords(text)
    print('processed_text:', processed_text)
    vectorized_text = vectorizer.transform([processed_text])
    similarities = cosine_similarity(vectorized_text, X)
    print('similarities:',similarities)
    max_similarities = np.max(similarities)
    print('max_similarities:',max_similarities)
    if max_similarities > 0.6:
        high_similarities_questions = [q for q, s in zip(question_list, similarities[0]) if s > 0.6]
        print('high_similarities_questions:', high_similarities_questions)
        
        target_answers = []
        for q in high_similarities_questions:
            q_index = question_list.index(q)
            target_answers.append(answer_list[q_index])
        print(target_answers)
        
        # Use the same vectorizer for both input and high similarity questions
        Z = vectorizer.transform([preprocess_with_stopwords(q) for q in high_similarities_questions])
        processed_with_stopwords = preprocess_with_stopwords(text)
        print('processed_with_stopwords:',processed_with_stopwords)
        vectorized_text_with_stopwords = vectorizer.transform([processed_with_stopwords])
        final_similarities = cosine_similarity(vectorized_text_with_stopwords, Z)
        closet = np.argmax(final_similarities)
        return target_answers[closet]
    else:
        return "I can't answer this Question"

get_response("are you right-handed?")


st.title("Simple Question-Answering Chat Bot")

user_input = st.text_input("Ask a question:")
if user_input:
    response = get_response(user_input)
    st.text("Response:")
    st.write(response)




