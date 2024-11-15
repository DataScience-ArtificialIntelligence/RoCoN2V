# -*- coding: utf-8 -*-

'''
Dependencies
!conda install tensorflow
!pip install ftfy==5.1
!conda install -c conda-forge spacy
!python -m spacy download en
!pip install tensorboardX
!pip install tqdm
!pip install pandas
!pip install ipython
!pip install nltk
'''

import nltk
import numpy as np
import pickle
from comet.csk_feature_extract import CSKFeatureExtractor


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def read_data(tran_path):
  f = open(train_path, 'r', encoding='utf8')
  dataset = f.readlines()
  f.close()
  Utterances = {}
  Utterance = []
  j =0
  for i, data in enumerate(dataset):
    if i < 2:
      continue
    if data == '\n':
      Utterances[j] = Utterance
      Utterance = []
      j +=1
    else:
      speaker, utt, emo, senti = data.strip().split('\t')
      Utterance.append(utt)
  return Utterances

Utterances = {}
train_path = 'train.txt'
Utterances = read_data(train_path)


count1 =0
for i in range(len(Utterances)):
  count1 += len(Utterances[i])


with open('meld_utterancedCOMPM.pkl', 'wb') as file:
  pickle.dump(Utterances, file)

extractor = CSKFeatureExtractor()
feaures = extractor.extract(Utterances)

with open('comet.pkl', 'rb') as file:
  comet = pickle.load(file)

count = 0
for key in comet[0].keys():
  count += len(comet[0][key])

comet_flatten = np.array([np.array(row) for conv in comet[0].keys() for row in conv])

with open('comet.pkl', 'wb') as file:
  pickle.dump(feaures, file)

for dataset in ['meld']:
    print ('Extracting features in', dataset)
    sentences = pickle.load(open(dataset + '/' + dataset + '_sentences.pkl', 'rb'))
    print(sentences.keys())
    break

extractor = CSKFeatureExtractor()

for dataset in ['meld']:
    print ('Extracting features in', dataset)
    sentences = pickle.load(open(dataset + '/' + dataset + '_sentences.pkl', 'rb'))
    feaures = extractor.extract(sentences)
    pickle.dump(feaures, open(dataset + '/' + dataset + '_features_comet.pkl', 'wb'))

print ('Done!')
