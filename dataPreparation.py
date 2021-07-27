from math import nan
import pickle
import pandas as pd
from future.utils import iteritems
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
tf.random.set_seed(123)
import numpy as np
from numpy import asarray, zeros
import math
import random
import itertools
np.random.seed(123)
random.seed(123)

class dataMaker(object):
    def __init__(self, config):
        self.config = config
        if self.config['Choice'] == 1:
            if not os.path.isfile(self.config['Path_to_Pickled_data_TrainTest']):
                self.trainSentences, self.testSentences = self.createTrainTest()
            else:
                pass
            
        elif self.config['Choice'] == 2:
            if not os.path.isfile(self.config['Path_to_Pickled_data_full']):
                self.trainSentences= self.createTrain() 
            else:
                pass
            
        else:
            pass   
    def makeSentences(self):
        df = pd.read_csv(self.config['Path_to_labeled_data'],sep='\t' ,encoding='utf-8',error_bad_lines=True,keep_default_na=False)
        data = df[['sentence_idx','word','tag']]
        agg_func = lambda s: [(w,t) for w,t in zip (s['word'].values.tolist(),s['tag'].values.tolist())]
        grouped = data.groupby('sentence_idx').apply(agg_func)
        sentences = [s for s in grouped]
        sentences.sort()
        sentences = list(sentences for sentences,_ in itertools.groupby(sentences))
        print('Total number of sentences in our dataset: ', len(sentences))
        return sentences
           
    def createTrainTest(self):
        sentences = self.makeSentences()
        testSentences = random.sample(sentences, k=math.floor(self.config['split_ratio'] * len(sentences)))
        trainSentences = [x for x in sentences if x not in testSentences]
        return trainSentences, testSentences
    def createTrain(self):
        sentences = self.makeSentences()
        return sentences
    def indexes(self):
        tags = set()
        words = set()
        for sent in self.trainSentences:
            for item in sent:
                tags.add(item[1])
                words.add(item[0].lower())
                
        tags = list(tags)
        words = list(words)
        tags.sort()
        tag2idx = {t:i for i,t in enumerate(tags)}
        idx2tag = {v:k for k, v in tag2idx.items()}  
        n_words = len(words)
        word2idx = {w:i+1 for i, w in enumerate(words)}
        word2idx.update({'UNK':0})
        word2idx.update({'__pad__':len(words)+1})
        idx2word = {i:w for w,i in word2idx.items()}
        maxlen = max([len(s) for s in self.trainSentences]) 
        if self.config['Choice'] == 1:
            index_obj = {'word2idx':word2idx,'idx2word':idx2word, 'tag2idx':tag2idx, 'idx2tag':idx2tag , 'n_words':n_words}
            with open(self.config['Path_to_Pickled_Idx'], 'wb') as f:
                pickle.dump(index_obj, f)
        else:
            index_obj = {'word2idx':word2idx,'idx2word':idx2word, 'tag2idx':tag2idx, 'idx2tag':idx2tag , 'n_words':n_words}
            with open(self.config['Path_to_Pickled_Idx_full'], 'wb') as f:
                pickle.dump(index_obj, f)
        return word2idx,idx2word, tag2idx, idx2tag , n_words, maxlen
        
    def processPretrainedWE(self, word2idx):
        embedding_index = {}
        with open(self.config['path_to_pretrained_WE'],'r') as f:
            for line in f.readlines()[1:]:
                values = line.split() 
                word = values[0] 
                coefs = asarray(values[1:], dtype='float32')
                embedding_index[word] = coefs 
        embedding_dim = len(values[1:])
        embedding_matrix = zeros((len(word2idx), embedding_dim))
        for word in word2idx:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[word2idx[word]] = embedding_vector
        return embedding_matrix
    
    @staticmethod
    def loadPickledIndecies(path):
        with open(path, 'rb') as f:
            idx_obj = pickle.load(f)
        word2idx, idx2word, tag2idx, idx2tag , n_words = idx_obj['word2idx'], idx_obj['idx2word'], idx_obj['tag2idx'], idx_obj['idx2tag'] , idx_obj['n_words']   
        return word2idx, idx2word, tag2idx, idx2tag , n_words
    @staticmethod
    def loadPickledData(path):
        with open(path, 'rb') as f:
            data_obj = pickle.load(f)
        return data_obj
    
    def makeData(self,word2idx, tag2idx):  
        if self.config['Choice'] == 1:
            X = [[word2idx[w[0].lower()] for w in s] for s in self.trainSentences]
            X_train = pad_sequences(maxlen= self.config['maxlen'], sequences= X, padding='post',truncating='post',value = word2idx['__pad__'])
            y = [[tag2idx[w[1]] for w in s] for s in self.trainSentences]
            y_train = pad_sequences(maxlen=self.config['maxlen'], sequences= y, padding='post',truncating='post',value=tag2idx['O'])
            
            X = [[word2idx[w[0].lower()] if w[0].lower() in word2idx else word2idx['UNK'] for w in s  ] for s in self.testSentences]
            X_test = pad_sequences(maxlen= self.config['maxlen'], sequences= X, padding='post',truncating='post',value = word2idx['__pad__'])
            y = [[tag2idx[w[1]] for w in s] for s in self.testSentences]
            y_test = pad_sequences(maxlen=self.config['maxlen'], sequences= y, padding='post',truncating='post',value=tag2idx['O'])
            data_obj = {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}
            with open(self.config['Path_to_Pickled_data_TrainTest'], 'wb') as f:
                pickle.dump(data_obj, f)
            return X_train, X_test, y_train, y_test
        
        if self.config['Choice']==2:
            X = [[word2idx[w[0].lower()] for w in s] for s in self.trainSentences]
            X_train = pad_sequences(maxlen= self.config['maxlen'], sequences= X, padding='post',truncating='post',value = word2idx['__pad__'])
            y = [[tag2idx[w[1]] for w in s] for s in self.trainSentences]
            y_train = pad_sequences(maxlen=self.config['maxlen'], sequences= y, padding='post',truncating='post',value=tag2idx['O'])
            data_obj = {'X_train':X_train, 'y_train':y_train}
            with open(self.config['Path_to_Pickled_data_full'], 'wb') as f:
                pickle.dump(data_obj, f)
            return X_train,  y_train
    
def outputPretify(TestSentences, PredictedEntities,word2idx,idx2word):
    SentLen = len(TestSentences[0])
    newSentences = []
    newPredictedEntities = []
    i = 0
    while(i<len(PredictedEntities)):
        padCount = list(TestSentences[i]).count(word2idx['__pad__'])
        newPredictedEntities.append(PredictedEntities[i][:SentLen-padCount])
        newSentences.append(TestSentences[i][:SentLen-padCount])
        i += 1
    newSentences = [[idx2word[i] for i in sent] for sent in newSentences]
    return  newSentences, newPredictedEntities


