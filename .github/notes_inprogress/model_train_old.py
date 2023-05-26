# data processing tools
import string, os 
import pandas as pd
import numpy as np
np.random.seed(42)
# to be able to use utils from the folder 
import sys
#sys.path.append("utils")
#import langutils as utils
#from langutils import clean_text
#from langutils import get_sequence_of_tokens
#from langutils import generate_padded_sequences
#from langutils import create_model
import tensorflow as tf
tf.random.set_seed(42)
# allow for random sampling 
from random import sample
# tensorflow packages
# keras module for building LSTM 
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
# saving models 
from joblib import dump
import pickle
# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


########## Ross' functions ############
def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower() # return vocabulary in the text if not part of string.punctuation and make lowercase 
    txt = txt.encode("utf8").decode("ascii",'ignore') # make sure utf8 encoding 
    return txt 

def get_sequence_of_tokens(tokenizer, corpus):
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences

def generate_padded_sequences(input_sequences, total_words):
    # get the length of the longest sequence
    max_sequence_len = max([len(x) for x in input_sequences])
    # make every sequence the length of the longest on
    input_sequences = np.array(pad_sequences(input_sequences, 
                                            maxlen=max_sequence_len, 
                                            padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, 
                            num_classes=total_words)
    return predictors, label, max_sequence_len

def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 
                        10, 
                        input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    model.add(Dropout(0.1)) # when determining weights during training, drop 10% of the weights - makes the model learn better, makes it a bit more difficult so it doesn't overfit as easily 
   
    # Add Output Layer
    model.add(Dense(total_words, 
                    activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam')
    
    return model

###################################

######### My script ##############

def load_data():
    # define file path
    data_dir = os.path.join("data", "news_data")
    all_comments = []
    for filename in os.listdir(data_dir):
        if 'Comments' in filename:
            comments_df = pd.read_csv(data_dir +  "/" + filename)
            all_comments.extend(list(comments_df["commentBody"].values))
    #randomly sample from the corpus due to large size 
    all_corpus = sample(all_comments, 10000) # can be made smaller 
    # clean the text 
    corpus = [clean_text(x) for x in all_corpus]
    return corpus

def tokenize(corpus):
    # tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    # turn sequence into tokens 
    inp_sequences = get_sequence_of_tokens(tokenizer, corpus)
    #pad input sequences 
    predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences, total_words)
    return predictors, label, max_sequence_len, total_words, tokenizer

def train_model(max_sequence_len, total_words, predictors, label):
    # train the model
    model = create_model(max_sequence_len, total_words)
    history = model.fit(predictors, 
                        label,
                        epochs=10, # can be made smaller 
                        batch_size=500, # picked this size due to how long it takes 
                        verbose=1)
    return model, history

def save_variables(model, tokenizer, max_sequence_len):
    # save the nodel
    file_path = r'models/model.pkl'
    pickle.dump(model, open(file_path, 'wb'))
    # save tokenizer
    dump(tokenizer, os.path.join("models", "Tokenizer.joblib"))
    # save max_sequence_len
    with open(os.path.join("models", "max_sequence_len.txt"), "w") as f:
        f.write(str(max_sequence_len))


def main():
    # load the data
    corpus = load_data()
    # tokenize the data
    predictors, label, max_sequence_len, total_words, tokenizer = tokenize(corpus)
    # create and train the model
    model, history = train_model(max_sequence_len, total_words, predictors, label)
    # saving the model, tokenizer, and max_sequence_len to be used in another script 
    save_variables(model, tokenizer, max_sequence_len)
    


if __name__=="__main__":
    main()

