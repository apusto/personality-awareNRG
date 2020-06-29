#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 18:31:47 2020

@author: annapustova
"""
import pandas as pd
import re
import glob
import os
import json
import random

dir_name = ".../KK_subtitles/"


# remove meta-information, lowercase, and strip punctuation marks

rm_metadata = [('\(.+\)', ''), ('"\\n(\d+;){1,}"', ' ')]
rm_punct = [('\n', ' '),(',', ' ,'),('.', ' .'), ('?', ' ?'), ('!', ' !'), ("♪",''), ("\\", ""), ('""','  ')]

def prepro(turn, repl=rm_punct, rm_metadata=rm_metadata):
    for exp, r in rm_metadata:
        turn = re.sub(exp, r, turn).strip()
    for t in repl:
        turn=turn.replace(*t)
    return turn.lower().strip()

kk_convturns = []

for filename in glob.iglob(".../KK_subtitles/" + '**/*', recursive=True):
    if filename.endswith(".csv"): 
        f = open(filename, 'r')
        turn = f.read()
        lines = turn.replace('...','.').split('>>')
        lines.pop(0)
        prepro_entries = [prepro(s, rm_punct,rm_metadata) for s in lines]
        dialogue= []
        prev_pers, prev_turn = None, None
        for idx,line in enumerate(prepro_entries):
            try:
                person, turn = line.split(":")
                if person == 'kim':
                    dialogue.append((prev_pers,prev_turn))
                    dialogue.append((person, turn))
                prev_pers, prev_turn = person, turn
            except:
                #print(line)
                pass
        if dialogue:
            kk_convturns.append(dialogue)

# save short dialogues the KK-Persona bi-RNN-LSTM model

with open('.../kk-bi-RNN-LSTM_data.json', 'w') as outfile:
    json.dump(kk_convturns, outfile)


"Transformation for the KK-Transformer model according to Wolf et al. (2019)"

facts = ["my name is kim kardashian .",
    "i was born on october 21 , 1980 , in los angeles , california . ",
                    "i have an older sister , kourtney , a younger sister , khloe , and a younger brother , rob .",
                    "after my parents divorced in 1991 , my mother married again that year , to caitlyn jenner .",
                    "i have step brothers burton , brandon , and brody and a step sister casey and half sisters kendall and kylie jenner .",
                    "i attended marymount high school , a roman catholic all-girls school in los angeles .",
                    "my father died in 2003 of cancer .",
                    "in my 20s, i was a close friend and stylist of socialite Paris Hilton .",
                    "in may 2014 i got married with kanye west .",
                    "i have four kids ."]


facts_prepro = [prepro(s) for s in facts]

# select turns form other than kim personas

def candidate_responses(num_candidates=7):
    candidate_response = []
    for _ in range(num_candidate_response):
        line = random.choice(random.choice(kk_convturns))
        while line[0] != 'kim':
            line = random.choice(random.choice(kk_convturns))
        candidate_response.append(line[1])
    return candidate_response


max_hist_len = 25
turns = []
utterance = []
for dialogue_id,dialogue in enumerate(kk_convturns):
    entry = {}
    history = []
    candidates = []
    utterance = []
    for line_id, line in enumerate(dialog):
        if line_id % 2 == 0:
            history.append(line[1])
            try:
                candidates = candidate_responses()
                candidates.append(line[1])
                #history.append(line[1])
                history = [x for x in history if x is not None and len(x)<511]
                candidates = [x for x in candidates if x is not None and len(x)<511]
                utterance.append({"candidates":candidates, "history":history.copy()})
                if len(history)>max_hist_len:
                    entry['personality'] = facts_prepro
                    entry['utterances'] = utterance
                    turns.append(entry)
                    entry = {}
                    history = []
                    candidates = []
                    utterance = []
            except Exception as e:
                print(e)
        else:
            history.append(line[1])
       
    entry['personality'] = facts_prepro
    entry['utterances'] = utterance
    turns.append(entry)

# split into train and validation sets
test_split = 0.2
ind = int(test_split*len(turns))

DATA = {'valid':turns[:ind], 'train':turns[ind:]}


with open('.../kk_transformer_data.json', 'w') as outfile:
    json.dump(DATA, outfile)
