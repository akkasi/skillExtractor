import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import contractions
import emoji
stopwords = stopwords.words('english')
tokenizer = RegexpTokenizer('[\$@]{,1}\w+[#@$+]{,2}\w*|[\$,;:.\-//\(\)]{1}')
patterns = [ r'•',
            r'http\S+',
            r'"',
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec)?[.,;]?\s*\d{1,2}[,;]?\s*(\d{4}|\d{2})[,;.]\s*\d{1,2}:\d{1,2}:\d{1,2} \w{2}\s*",
            r"\d{1,2}[,.;]\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec)?[.,;]?\s*(\d{4}|\d{2})[,;.]?\s*\d{1,2}:\d{1,2}:\d{1,2} \w{2}\s*", 
            r"\d{1,2}:\d{1,2}:\d{1,2} \w{0,2}\s*",
            r"\[[^]]*\]"
           ]
def rmStopWords(Sentence):
    S = ' '.join([s for s in Sentence.split() if s.lower() not in stopwords])
    return S
def preprocessing(String):
    p = re.compile('\n{2,}')
    text = re.sub(p,'\n',str(String))
    text = emoji.get_emoji_regexp().sub(u'', text)
    text = re.sub(' +',' ',text)
    text = contractions.fix(text)
    for p in patterns:
        text = re.sub(p,'',text)
    return text.strip()

def padSequence(texts,word2idx,maxlen):
    X = [[word2idx[w.lower()] if w in word2idx else word2idx['UNK'] for w in s] for s in texts]
    X = pad_sequences(maxlen= maxlen, sequences= X, padding='post',truncating='post', value = word2idx['__pad__'])
    return X
def newTextPrep(text,word2idx,maxlen):
    text = preprocessing(text)
    lines = text.splitlines()
    sentences = []
    for l in lines:
        sentences.extend(sent_tokenize(l))
    
    sentences = [s if s[-1] not in ['.',',',';','/','-','(',')','[',']','%','^','@','!','^','&','\\'] else s[:-1] for s in sentences]
    texts = [tokenizer.tokenize(l.lower()) for l in sentences]
    
    return padSequence(texts,word2idx,maxlen)
def SkillPresenter(List):
    Full = []
    Partial = []
    for x in List:
        i = 0
        while i < len(x):
            t = []
            b = []
            if x[i][1].startswith('B-'):
                t.append(x[i][0])
                j=i+1
                while j<len(x):
                    if x[j][1].startswith('I-'):                
                        t.append(x[j][0])
                        
                        j+=1
                    break
                Full.append(t)        
                i = j
            
            elif x[i][1].startswith('I-'):
                b.append(x[i][0])
                j=i+1
                while j<len(x):
                    if x[j][1].startswith('I-'):                
                        b.append(x[j][0])
                        
                        j+=1
                    break
                Partial.append(b)        
                i = j
            else:
                i+=1
    Full = [' '.join(i) for i in Full]
    Partial = [' '.join(i) for i in Partial]
    return Full, Partial

def showResults(Input,Predictions):
    Results = []
    i= 0
    while i< len(Predictions):
        Results.append([(token,Tag) for token, Tag in zip(Input[i], Predictions[i])])
        i+=1
    Full, Partial = SkillPresenter(Results)
    return Full, Partial    


# 