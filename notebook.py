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
data_meta.head(20)


# output as Python script for nice version-controlling
get_ipython().system('jupyter nbonvert --to script notebook.ipynb')

