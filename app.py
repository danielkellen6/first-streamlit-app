#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DanKellen@ MABA CLASS
"""

import streamlit as st
import pandas as pd
import string
import plotly.express as px
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import os
from spacy import displacy
from sentence_transformers import SentenceTransformer
import scipy.spatial
import pickle as pkl
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import nltk
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
# from spacy import en_core_web_sm

embedder = SentenceTransformer('all-MiniLM-L6-v2')
# nlp = spacy.load("en_core_web_sm")


header = st.container()
dataset = st.container()
search = st.container()


with header:
    st.title("Miami Hotel Search")
    st.markdown("As Will Smith so eloquently stated: *Welcome to Miami - Bienvenidos a Miami*")
    st.markdown("Use this to search to find the best Miami hotel to fit your needs.")


with dataset:
    #st.header('Miami Hotels List')
    df = pd.read_csv("hotelReviewsInMiami__en2019100120191005.csv")
    hotel_list = pd.read_csv("HotelListInMiami__en2019100120191005.csv")
    #st.write(df.head())
    punctuation = string.punctuation
    def remove_punctuation(text):
        no_punct=[words for words in text if words not in punctuation]
        words_wo_punct=''.join(no_punct)
        return words_wo_punct

    def first_line(text):
        hotel_name = text.partition('\n')[0]
        hotel_name = hotel_name[2:]
        return hotel_name

    digits = string.digits
    def remove_digits(text):
        no_nums=[words for words in text if words not in digits]
        words_wo_nums=''.join(no_nums)
        return words_wo_nums
    def lower(text):
        all_lower=[words.lower() for words in text]
        words_all_lower =''.join(all_lower)
        return words_all_lower

    df['DispHotel']=df['hotelName'].apply(lambda x: first_line(x))
    df['hotelName']=df['hotelName'].apply(lambda x: remove_punctuation(x))
    df['hotelName']=df['hotelName'].apply(lambda x: first_line(x))
    df['hotelName']=df['hotelName'].apply(lambda x: remove_digits(x))
    df['review_body']=df['review_body'].apply(lambda x: lower(x))

    #st.write(hotel_list['hotel_name'], index = False)

    df_combined = df.sort_values(['hotelName']).groupby('hotelName', sort=False).review_body.apply(''.join).reset_index(name='review_body')
    #st.write(df_combined.head())


    def lsa_method(text):
      parser = PlaintextParser.from_string(text, Tokenizer("english"))
      summarizer_lsa = LsaSummarizer()
      summary_2 = summarizer_lsa(parser.document, 2)
      dp = []
      for i in summary_2:
        lp = str(i)
        dp.append(lp)
        final_sentence = '  '.join(dp)
      return final_sentence

    df_combined['Review_Summary']=df_combined['review_body'].apply(lambda x: lsa_method(x))
    df2 = df_combined
    df_sentences = df_combined.set_index("review_body")
    df_sentences = df_sentences["hotelName"].to_dict()
    df_sentences_list = list(df_sentences.keys())
    df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]
    corpus = df_sentences_list
    corpus_embeddings = embedder.encode(corpus,show_progress_bar=False)
    model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
    embeddings = model.encode(corpus)


with search:
    queries = st.text_input("What are you looking for in a hotel?",'Search')

    query_embeddings = embedder.encode(queries,show_progress_bar=False)
    top_k = min(1, len(corpus))
    for query in queries:

        query_embedding = model.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        for score, idx in zip(top_results[0], top_results[1]):
            #print(corpus[idx], "(Score: {:.4f})".format(score))
            row_dict = df2.loc[df2['review_body'] == corpus[idx]]
            Hotel = " ".join(row_dict['hotelName'])
            Summary = " ".join(row_dict['Review_Summary'])


    st.header("The best hotel for your stay: ")
    st.write(Hotel)
    st.header("Summary reviews from previous guests:")
    st.write(Summary)


    # try:
    #     from googlesearch import search
    # except ImportError:
    #     print("No module named 'google' found")
    #
    # # to search
    # hotel_link = search(Hotel, tld="co.in", num=1, stop=1, pause=2)


    from googlesearch import search
    for url in search(Hotel, stop=1):
        url_link = url
    st.header("Book your stay today!")
    st.write("[Click here to be redirected to the Hotel Webpage](%s)" % url_link)


# @st.cache(persist=True)
# def load_data():
#     df = pd.read_csv("HotelListInMiami__en2019100120191005.csv")
#     return(df)
#
#
#
# def run():
#
#
#     df = load_data()
#
#     if disp_head=="Head":
#         st.dataframe(df.head())
#     else:
#         st.dataframe(df)
#
#
#     #Add images
#     #images = ["<image_url>"]
#     #st.image(images, width=600,use_container_width=True, caption=["Iris Flower"])
#
#



# if __name__ == '__main__':
#     run()
