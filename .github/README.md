
# Assignment 3 - Language modelling and text generation using RNNs

This assignment can be found at my github [repo](https://github.com/AU-CDS/assignment-3---rnns-for-text-generation-ameerwald). This will be updated later when submitting so that it is within my own personal repo and not in the github classroom. 

In this assignemnt, we were asked to create two scripts using ```TensorFlow``` to build complex deep learning models for NLP. The first script will train and save a text generation model on some culturally significant data - comments on articles for *The New York Times*. You can find the data [here](https://www.kaggle.com/datasets/aashita/nyt-comments). The second script will use that model to generate new text from a user-suggested prompt. 


# Repository 

| Folder         | Description          
| ------------- |:-------------:
| In      | Data - normally the data would be here but it is too large to push to Github so it can be retrived from the link above 
| Models  | Saved the model plus tokenizer and max_sequence_len txt file for using the model 
| Notes | Jupyter notebook and script with notes       
| Out  |  A text file of the generated text   
| Src  | Two py scrips, 1) creating and training a model 2) generating new text based on the model    
| Utils  | Utilities script - functions we were given in class       


## To run the scripts 

From the command line, run the following chunk of code.  
``` 
bash setup.sh
run.sh
```

If issues occur regarding setting up the virtual environment, run the following two lines of code first in the command line. 
```
sudo apt-get update
sudo apt-get install python3-venv
```
Due to the computational complexity of the task, it took hours to run as is. To cut down on time the following two lines in the ```model_train.py``` script can be adjusted.  
Line 98, the number of comments can be made smaller. 
```
all_corpus = sample(all_comments, 10000) 
```

Line 119, in the train_model function, the number of epochs can be cut down as well as adjusting the batch size.

When testing the scripts, I used 500 comments and 1 epoch, which ran for less than 30 minutes but generated very poor text. When run as is, the text generated is much better but as mentioned, takes 3-5 hours to run. 
