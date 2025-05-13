# Natural Language Programming: N-Grams
## Overview
This project was to develop a simple n-gram algorithm. It reads in a corpus of text to predict what the next word should be given an input.

- The [Blogs](Blogs/) folder contains the files with the text to train on. They are formatted as HTML in .xml files. Only the text within the <post> tags are used for training.
- The [ngram_functions.py]() file contains the functions to perform the algorithm
- The [main.ipynb]() file shows the code being run in a Jupyter Notebook and the results on a few small sample texts.

## Data Processing

The data is given as separate xml files in a folder. Only the text within the \<post> tags are used for training so those are extracted. There is no reason for the relevant text to be separated based on file/post so all relevant training text is combined together after being stripped (no extra whitespaces, extra puncuation, etc.) and case-folded (all characters but in lower-case). 

Sentence markers (periods, exclamation marks, etc.) are left in and there are normal spaces between words. Sentences are separated so that beginning and end sentence tokens can be added. Also, the words in the sentences are tokenized as well (ex: contractions expanded out). 

## Algorithm
### Training
Takes in the training text inputs and gets the counts for:
- 1-Grams: The number of times each token appear in the whole training corpus.
- 2-Grams: The number of times each combination of 2 tokens (order matters) appear in the whole training corpus.
- 3-Grams: The number of times each combination of 3 tokens (order matters) appear in the whole training corpus.

### Prediction
A prediction for this algoritm will take some input text and predict what the next token will be.
- 1-Grams: Doesn't actually care about the input text, will always predict what the most common token in the training was.
- 2-Grams: Predict the next token based on the last token from the input text.
- 3-Grams: Predict the next token based on the last 2 tokens from the input text.

See [ngram_functions.py](ngram_functions.py#L143-L157) for the specific formula being used.
