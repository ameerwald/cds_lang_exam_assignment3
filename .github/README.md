
# Assignment 3 - Language modelling and text generation using RNNs

## Github repo link 

This assignment can be found at my github [repo](https://github.com/ameerwald/cds_lang_exam_assignment3). 

## The data

A dataset of comments on articles from *The New York Times* is used in this assignment. You can find the data [here](https://www.kaggle.com/datasets/aashita/nyt-comments). 

## Assignment description

In this assignment, we were asked to create two scripts using ```TensorFlow``` to build complex deep learning models for NLP. The first script will train and save a text generation model on some culturally significant data from *The New York Times*. The second script will use that model to generate new text from a user-suggested prompt. 


# Repository 

| Folder         | Description          
| ------------- |:-------------:
| In      | This folder is hidden due to the size of the dataset  
| Models  | Saved the model plus tokenizer and max_sequence_len txt file for using the model 
| Notes | Jupyter notebook and script with notes       
| Out  |  A text file of the generated text   
| Src  | Two py scrips, 1) creating and training a model 2) generating new text based on the model    
| Utils  | Utilities script - functions used in src scripts      


## To run the scripts 

As the dataset is too large to store in my repo, use the link above to access the data. Download and unzip the data. Then create a folder called  ```in``` within the assignment 3 folder, along with the other folders in the repo. Then the code will run without making any changes. If the data is placed elsewhere, then the path should be updated in the code.

1. Clone the repository, either on ucloud or something like worker2
2. From the command line, at the /cds_vis_exam_assignment1/ folder level, run the following lines of code. 

This will create a virtual environment, install the correct requirements.
``` 
bash setup.sh
```
While this will run the scripts and deactivate the virtual environment when it is done. 
```
bash run.sh
```

Due to the computational complexity of the task, it took hours to run in order to perform "well". To cut down on time the following two lines in the ```model_train.py``` script can be adjusted.  
Line 42, the number of comments can be made smaller. 
```
all_corpus = sample(all_comments, 10000) 
```

Line 63, in the train_model function, the number of epochs can be adjusted lower to perform faster or higher to perform better.  
```
epochs=2, 
```

To personalize the text results. In the ```generate_text.py``` script the following code, on line 55 can be adjusted.
```
new_text = generate_text("trump", 10, model, max_sequence_len, tokenizer)
``` 
A new keyword can be entered in quotes instead of "trump" and the number 10 can be adjusted - it stipulates the number of words in the generated text. 

This has been tested on an ubuntu system on ucloud and therefore could have issues when run another way.

## Discussion of Results

When testing the scripts, I used 500 comments and 1 epoch, which ran for less than 30 minutes but generated very poor text. When run with 10000 comments and 10 epochs, the text generated is much better but takes 3-5 hours to run even on a 64 CPU machine. 

I have currently set it to run with 1000 comments over 10 epochs which generated the following text "Trump The The Country Of The Country Of The Country Of". Clearly not great but it has learned some. As mentioned, with more computational power and time the results improve but for the sake of this assignment that did not feel necessary.  
