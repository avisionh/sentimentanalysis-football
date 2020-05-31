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

# We now reduce the dataset we are interested in so that it's easier to handle throughout. Note that the `event_team` column is defined as the team that produced the event. Thus, the commentary, `text` column, is associated to this column. In the case of an own goal, then the `event_team` will be the team that benefitted from the goal. In which case, we need to consider this carefully because the commentary associated to an own goal would be negative, but the team associated to the commentary is the one that benefits, hence we need to treat these cases differently.
# 
# Note, we can identify own goals via the `event_type2` column which will have a value of `15`. Thus, we need to switch the entries around for `event_team` and `opponent` for these cases.

# select only necessary columns
data_commentary = data_commentary.loc[:, ['id_odsp', 'text', 'event_team', 'opponent']]
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
#     
# Things worth considering also:
# 
# - Commentary for a game will be positive for one side and negative for the other side.
#     + This is a loose assumption as the losing team can perform well. In which case, the winning and losing side can both have postive sentiment. This should be okay because good teams that lose are still good teams. (This could be the unexplained 'luck' factor)
# - A team may get negative sentiment for sitting back, soaking up pressure, hitting the opponent on the counterattack and winning the game.
#      + In which case, are we judging teams that predominantly counterattack as bad? Is this desirable? A good counter-example would be Leicester City in the 2015-16 EPL season when they played a predominantly counter-attacking game but won the league. Clearly they must have been a good side to win the league, so we would not want to judge them as being a bad team.

aggregate_function = {'text': ' '.join}
data_reduced = data_commentary.groupby(['id_odsp', 'event_team', 'opponent']).agg(aggregate_function)


data_reduced.head(10)


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




