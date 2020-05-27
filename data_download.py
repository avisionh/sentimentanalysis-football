#!/usr/bin/env python
# coding: utf-8

# https://technowhisp.com/kaggle-api-python-documentation/
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi


# initialise and authenticate
api = KaggleApi()
api.authenticate()

# list dataset files
api.dataset_list_files(dataset = 'secareanualin/football-events').files

# download all files
api.dataset_download_files(dataset = 'secareanualin/football-events', path = 'data')

# unzip file
with zipfile.ZipFile('data/football-events.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

# delete zip file
os.remove('data/football-events.zip')

