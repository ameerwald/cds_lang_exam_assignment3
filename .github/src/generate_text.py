# data processing tools
import string, os 
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
tf.random.set_seed(42)
# import packages for model and needed variables
from joblib import load
import pickle



def load_model():
  # loading the model 
  file_path = r'models/model.pkl'
  model = pickle.load(open(file_path,'rb'))
  # loading the tokenizer 
  tokenizer = load("models/Tokenizer.joblib")
  # load max_sequence_len
  with open("models/max_sequence_len.txt") as f: 
    lines = f.readlines()  
  # making this into an int to be used later 
  for line in lines: 
    # return a list of strings in the file (in this case only though)
    words = line.strip("\n").split() 
    # turn that list of strings into a list of ints
    int = [eval(x) for x in words]
    # take the first one and only one in this case 
    max_sequence_len = int[0]
  return model, tokenizer, max_sequence_len


def generate_text(seed_text, next_words, model, max_sequence_len, tokenizer): 
    for _ in range(next_words): 
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], 
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


def main():
    # load the model and variables
    model, tokenizer, max_sequence_len = load_model()
    # generate new text 
    new_text = generate_text("trump", 10, model, max_sequence_len, tokenizer)
     # save the generated text  
    with open(os.path.join("out", "generated_text.txt"), "w") as f:
        f.write(new_text)
    


if __name__=="__main__":
    main()

