import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn_crfsuite.metrics import flat_classification_report
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense, LSTM, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from tf2crf import CRF, ModelWithCRFLoss
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import datetime
plt.style.use('ggplot')
tf.random.set_seed(123)
np.random.seed(123)
class NER(object):
    def __init__(self,config,nWords, nTags,Embedding_weight):
        self.config = config
        self.embDim = self.config['WordEmbedding_dim']
        self.inDim = self.config['maxlen']
        self.dropout = self.config['dropout']
        self.nWords = nWords
        self.nTags = nTags
        self.Embedding_weight = Embedding_weight
        
    def creatModel(self):
        input = Input(shape=(self.inDim,))
        
        if self.Embedding_weight is not None:
            model = Embedding(input_dim=self.Embedding_weight.shape[0],
                              output_dim=self.Embedding_weight.shape[1],
                              input_length=self.inDim, weights = [self.Embedding_weight],
                              trainable=False)(input)
        else:
            print(1)
            model = Embedding(input_dim=self.nWords, output_dim=self.embDim,
                         input_length=self.inDim)(input)

        model = Bidirectional(LSTM(units=self.embDim,return_sequences=True,
                                    dropout=self.dropout, recurrent_dropout=self.dropout))(model)

        model = LSTM(units=self.embDim * 2, return_sequences=True,
                    dropout=self.dropout, recurrent_dropout=self.dropout)(model)
                    
        model = TimeDistributed(Dense(units=self.nTags,activation='relu'))(model)

        self.crf = CRF(self.nTags)
        out = self.crf(model)
        model = Model(input, out)
        model = ModelWithCRFLoss(model, sparse_target=True)
        model.build((None,self.inDim,))
        return model

    def trainModel(self,model,X_train,y_train):
        adam = optimizers.Adam(learning_rate=self.config['lr'])
        model.compile(optimizer=adam,metrics=[self.crf.accuracy_fn,'accuracy'])
        if self.config['Choice']==2:
            filepath=self.config['Path_to_Trained_models']+"NER-model-FullData-{val_val_accuracy:.2f}-"+datetime.datetime.today().strftime("%b-%d-%Y")+".hdf5"
        else:
            filepath=self.config['Path_to_Trained_models']+"NER-model-{val_val_accuracy:.2f}-"+datetime.datetime.today().strftime("%b-%d-%Y")+".hdf5"
             
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_val_accuracy',
                                        verbose=1, save_best_only=True,
                                        save_weights_only=True,
                                        mode='max')
        callbacks_list =[checkpoint]
        
        self.history = model.fit(X_train,np.array(y_train),batch_size=self.config['batch_size'],
                             epochs=self.config['epochs'], validation_split=self.config['validation_split'],
                             verbose=1, callbacks=callbacks_list)

    @staticmethod
    def loadModel(pathtoModel):
        return tf.keras.models.load_model(pathtoModel)
    def plot_history(self):
        history_dict = self.history.history
        print(history_dict.keys())
        accuracy = self.history.history['accuracy']
        val_accuracy = self.history.history['val_val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss_val']
        x = range(1, len(accuracy) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, accuracy, 'b', label='Training acc')
        plt.plot(x, val_accuracy, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend() 
        plt.show()

    @staticmethod
    def makePrediction(model, X_test):
        return model.predict(X_test, verbose=0)

    @staticmethod
    def pred2label(predictions,idx2tag):
        out = []
        for pred_i in predictions:
            out_i = []
            for p in pred_i:
                
                out_i.append(idx2tag[p])
            out.append(out_i)
        return out
    @staticmethod
    def evaluate(test_labels,pred_labels):
        print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
        print("Precision-score: {:.1%}".format(precision_score(test_labels, pred_labels)))
        print("Recall-score: {:.1%}".format(recall_score(test_labels, pred_labels)))
        print('************************')
        print('************************')
        report = flat_classification_report(y_pred=pred_labels, y_true=test_labels)
        print(report)







    


