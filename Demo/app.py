from random import choice
from flask import Flask, render_template, url_for,request
import numpy as np
import os,sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from dataPreparation import dataMaker, outputPretify
from models import NER
np.random.seed(123)
from newTextPreparation import *
import yaml
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
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
    

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process',methods=['POST'])
def process():
    if request.method == 'POST':
        choice = request.form['taskoption']
        rawtext = request.form['rawtext']
        
        X_test = newTextPrep(rawtext,word2idx,config['maxlen'])
        y_pred =  model.predict(X_test, verbose=0)
        pred_labels = NER.pred2label(y_pred,idx2tag)
        X_test, pred_labels = outputPretify (X_test,pred_labels,word2idx,idx2word)
        Full, Partial = showResults(X_test, pred_labels)
        
        if choice == 'Skills':
            results = list(set(Full)) + list(set(Partial))
            if 'UNK' in results:
                results.remove('UNK')
            results.sort()
            num_of_results = len(results)
        if choice == 'Occupations':
            pass
        if choice == 'Tasks':
            pass
        if choice == 'Knowledge':
            pass
        if choice == 'Abilities':
            pass
        if choice == 'Attitudes':
            pass
    return render_template('index.html', results=results, num_of_results=num_of_results)        
if __name__ == '__main__':
    app.run(debug=True)