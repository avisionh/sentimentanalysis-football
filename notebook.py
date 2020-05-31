#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis
# Using `textblob` library
# 
# Data comes from Kaggle dataset of football events from [Alin Secareanu](https://www.kaggle.com/secareanualin/football-events).

import pandas as pd
from textblob import TextBlob

# display multiple outputs in same cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


data_commentary = pd.read_csv(filepath_or_buffer = 'data/events.csv')
data_meta = pd.read_csv(filepath_or_buffer = 'data/dictionary.txt', sep = '\t')


# view data
data_commentary.head(10)
data_meta.head(15)


# check info of data types
data_commentary.info()


# Need to convert object of interest into a `TextBlob` object so we can use things from `TextBlob` package.

data_commentary['text'] = data_commentary['text'].astype(str)


# inspect sentiment of each row
for row_line in data_commentary['text']:
    row_analyse = TextBlob(row_line)
    row_analyse.sentiment


# output as Python script for nice version-controlling
get_ipython().system('jupyter nbconvert --to script --no-prompt notebook.ipynb')




