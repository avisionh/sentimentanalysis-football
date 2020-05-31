#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis
# Using `textblob` library
# 
# Data comes from Kaggle dataset of football events from [Alin Secareanu](https://www.kaggle.com/secareanualin/football-events).
# 
# Following the below tutorials:
# 
# - [Python programming](https://pythonprogramming.net/sentiment-analysis-python-textblob-vader/)
# - [DigitalOcean](https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk)

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


# Now, we could analyse the sentiment of each individual line of commentary but thinking about what we are trying to achieve, namely:
# 
# > Analyse the sentiment associated to teams for **each game** so that we have a proxy for how good the team is
# 
# Then given this aim, what we want to do is *squash* **all** the commentary associated to the home and away side in a game and analyse the overall sentiment for each side from that. We then have sentiments attached to each team for each game and by deciding the sentiments overall, we can gauge how well the team performed in all their games within the dataset and henceforth, how good the team are. 
# 
# For instance, consider Borussia Dortmund. If the overall sentiment from each of their games played was positive, and most of their games had a positive sentiment, then this suggests Borussia Dortmund are a good team, which as a football fan, we know is true. 
# 
# ### Assumption
# Within this framework, we make the following assumptions:
# 
# - Commentary recorded in the dataset reports on all main events in a game.
# - Commentary recorded in the dataset will report the good and bad events associated to the home and away sides in each game.
#     + This means the commentary is an accurate reflection of all the major events in each game.

# ***
# 
# ## Baseline
# In this section, we will apply the `.sentiment` method on our commentary so that we can get a baseline of results that we can then compare future outputs against. In particular, the idea is that we apply this `sentiment` method on the unprocessed data. Then we process the data by doing things like:
# 
# - Lemmatise
# - Stem
# - ...
# 
# After processing the data, we should get a better dataset to start extracting the sentiments from.

# inspect sentiment of each row
for row_line in data_commentary['text']:
    row_analyse = TextBlob(row_line)
    row_analyse.sentiment


# output as Python script for nice version-controlling
get_ipython().system('jupyter nbconvert --to script --no-prompt notebook.ipynb')




