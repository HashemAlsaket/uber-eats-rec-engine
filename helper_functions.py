from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re

import requests as req

from selenium import webdriver

import gensim.downloader as w2v_api


wv = w2v_api.load('word2vec-google-news-300')


def scrape_menu_page(url):
    driver = webdriver.Chrome(executable_path="chromedriver_win32/chromedriver.exe")
    driver.maximize_window()
    driver.set_page_load_timeout(30)
    try:
        driver.get(url)
    except:
        return ""
    
    menu_page = driver.page_source
    driver.close()
    
    return menu_page


def normalize_w2v(word_vec):
    norm = np.linalg.norm(word_vec)
    if norm == 0: 
        return word_vec
    return word_vec/norm


def avg_wv_bow(terms):
    denom = 0
    arr = [0] * 300
    
    for term in terms:
        if term not in wv:
            continue
        arr = np.add(arr, normalize_w2v(wv[term]))
        denom += 1
    return arr / denom