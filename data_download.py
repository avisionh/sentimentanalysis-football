#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://technowhisp.com/kaggle-api-python-documentation/
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi


# In[2]:


# initialise and authenticate
api = KaggleApi()
api.authenticate()


# In[4]:


# list dataset files
api.dataset_list_files(dataset = 'secareanualin/football-events').files


# In[7]:


# download all files
api.dataset_download_files(dataset = 'secareanualin/football-events', path = 'data')


# In[9]:


# unzip file
with zipfile.ZipFile('data/football-events.zip', 'r') as zip_ref:
    zip_ref.extractall('data')


# In[10]:


# delete zip file
os.remove('data/football-events.zip')


# In[ ]:


# convert notebook to HTML for easy version-control
get_ipython().system('jupyter nbconvert --to html data_download.ipynb')

