# data processing tools
import string, os 
import numpy as np
import pandas as pd
import tensorflow.keras.utils as ku 
# keras module for building LSTM 
import tensorflow as tf
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

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
    input_sequences = np.array(ku.pad_sequences(input_sequences, 
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
    model.add(Dropout(0.1)) 
   
    # Add Output Layer
    model.add(Dense(total_words, 
                    activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam')
    
    return model

def generate_text(seed_text, next_words, model, max_sequence_len): # seed_text like a prompt we give it 
    for _ in range(next_words): # for however many in the range of next words, like 5 
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = ku.pad_sequences([token_list], 
                                    maxlen=max_sequence_len-1, 
                                    padding='pre')
        predicted = np.argmax(model.predict(token_list),
                                            axis=1)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()


if __name__=="__main__":
    main()