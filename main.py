

import numpy as np
import os
from dataPreparation import dataMaker, outputPretify
from models import NER
np.random.seed(123)
from newTextPreparation import *
import yaml
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def main():
    if config['Choice'] == 1:
        data = dataMaker(config)
        if os.path.isfile(config['Path_to_Pickled_Idx']):
            word2idx, idx2word, tag2idx, idx2tag , _ = dataMaker.loadPickledIndecies(config['Path_to_Pickled_Idx'])
        else:
            word2idx, idx2word, tag2idx, idx2tag , _, _ = data.indexes()
        print('Number of unique words in training data: ', len(word2idx) )
        if  os.path.isfile(config['Path_to_Pickled_data_TrainTest']):
            data_obj = dataMaker.loadPickledData(config['Path_to_Pickled_data_TrainTest'])
            X_train, X_test, y_train, y_test  = data_obj['X_train'], data_obj['X_test'], data_obj['y_train'], data_obj['y_test']
        else:
            X_train, X_test, y_train, y_test = data.makeData(word2idx, tag2idx)
        print('number of training samples is: ', len(X_train))
        print('number of test samples is: ', len(X_test))
        if config['use_pretrainedWE']:
            Embedding_weight = data.processPretrainedWE(word2idx)
        else:
            Embedding_weight = None
        model_obj = NER(config, nWords = len(word2idx), nTags = len(tag2idx),Embedding_weight=Embedding_weight)
        model = model_obj.creatModel()
        model_obj.trainModel(model,X_train,y_train)
        test_pred = NER.makePrediction(model,X_test)
        pred_labels = NER.pred2label(test_pred,idx2tag)
        test_labels = NER.pred2label(y_test,idx2tag)
        NER.evaluate(test_labels,pred_labels)
        
    elif config['Choice'] == 2:
        data = dataMaker(config)
        if os.path.isfile(config['Path_to_Pickled_Idx_full']):
            word2idx, idx2word, tag2idx, idx2tag , _ = dataMaker.loadPickledIndecies(config['Path_to_Pickled_Idx_full'])
        else:
            word2idx, idx2word, tag2idx, idx2tag , _, _ = data.indexes()
        print('Number of unique words in training data: ', len(word2idx) )
        if  os.path.isfile(config['Path_to_Pickled_data_full']):
            data_obj = dataMaker.loadPickledData(config['Path_to_Pickled_data_full'])
            X_train, y_train = data_obj['X_train'], data_obj['y_train']
        else:
            X_train, y_train = data.makeData(word2idx, tag2idx)
        print('number of training samples is: ', len(X_train)) 
        if config['use_pretrainedWE']:
            Embedding_weight = data.processPretrainedWE(word2idx)
        else:
            Embedding_weight = None
    
        model_obj = NER(config, nWords = len(word2idx), nTags = len(tag2idx),Embedding_weight=Embedding_weight)
        model = model_obj.creatModel()
        model_obj.trainModel(model,X_train,y_train)
        
    elif config['Choice'] == 3:
        data = dataMaker(config)
        if os.path.isfile(config['Path_to_Pickled_Idx']): # it can also be 'Path_to_Pickled_Idx_full'
            word2idx, idx2word, tag2idx, idx2tag , _ = dataMaker.loadPickledIndecies(config['Path_to_Pickled_Idx'])
            if config['use_pretrainedWE']:
                Embedding_weight = data.processPretrainedWE(word2idx)
            else:
                Embedding_weight = None  
            model_obj = NER(config, nWords = len(word2idx), nTags = len(tag2idx),Embedding_weight=Embedding_weight)
            model = model_obj.creatModel()
            model.load_weights(config['path_to_Best_Trained_model'])
            while True:
                text = input("Please Enter the text you want to extract its entities: ")                       
                if text == 'exit':
                    break
                X_test = newTextPrep(text,word2idx,config['maxlen'])
                y_pred =  model.predict(X_test, verbose=1)
                pred_labels = NER.pred2label(y_pred,idx2tag)
                X_test, pred_labels = outputPretify (X_test,pred_labels,word2idx,idx2word)
                Full, Partial = showResults(X_test, pred_labels)
                print("Fully detected skills are: ", Full)
                print("Partially detected skills are: ", Partial)
        
        else:
            print('Check the indices pickle file!')
   
    else:
        print ('The choice is not valid!!')
    
if __name__== "__main__":
    main()
