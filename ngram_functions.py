import os

from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter, defaultdict


def read_blogs(blog_dir):
    '''
    Goes through each xml file in the directory and gets only the text contained within the <post> tags and combines them into one string.
    Ignores text in other tags and the tags themselves.

    :param blog_dir: A valid directory string for where the xml files are located.
    :return: Returns a string of every single post contained within all of the blog xml files.  
    '''

    file_list = os.listdir(blog_dir)
    combined_string = ''
    for file in file_list:
        print("Now reading " + file)
        # Open the file in read only mode
        with open(blog_dir + "\\" + file, 'r', encoding="ansi") as file_data:
            # Change the data to the Beautiful Soup type for easier xml text extraction from tags
            text = BeautifulSoup(file_data)
        # Get array containing each post's text
        posts = text.find_all("post")
        # Add all the text from each post into one string
        for post in posts:
            # Need + ' ' so that the end of a post and beginning of the next post are not combined into one word/token
            combined_string += post.get_text(strip=True).casefold() + ' '
    print("Done reading in data")
    return combined_string


def count_words(text):
    '''
    Goes through the text and properly breaks it into the sentences and properly breaks those into the words.
    Finds the number of times each combination of words appear between all of the sentences (single words, two word combinations and three word combinations).
    These word combinations are called n-grams where n is the number of words in the combination

    :param text: The full text to analyze as a string.
    :return: Returns the dictionaries for the 1-grams, 2-grams, and 3-grams in that order.
    '''

    print("Counting tokens please wait")
    sentences = sent_tokenize(text)
    all_tokens = []
    for sentence in sentences:
        # Goes through each sentence in the text and makes them lowercase and
        # tokenized with start/end of sentence symbols added
        sentence = sentence.casefold()
        sentence_tokens = word_tokenize(sentence)
        sentence_tokens.append("</s>")
        sentence_tokens.insert(0, "<s>")
        all_tokens.append(sentence_tokens)

    one_grams, two_grams, three_grams = defaultdict(int), defaultdict(int), defaultdict(int)
    for sentence in all_tokens:
        # Goes through each of the sentence's tokens and finds all 1/2/3-grams and
        # updates the count of that n-gram
        for i in range(len(sentence)):
            # Go through each sentence only once with a group of 3 words at a time
            word1 = sentence[i]
            word2 = sentence[i+1] if (i+1) <= (len(sentence)-1) else None
            word3 = sentence[i+2] if (i+2) <= (len(sentence)-1) else None

            one_grams[word1] += 1

            if word2:
                two_gram = " ".join([word1, word2])
                two_grams[two_gram] += 1
            
            if word3:
                three_gram = " ".join([word1, word2, word3])
                three_grams[three_gram] += 1
    
    print("Number of tokens: " + str(sum(one_grams.values())))
    print("Number of words: " + str(len(one_grams)))
    print("\n")
    return one_grams, two_grams, three_grams


def print_frequent_n_grams(unigrams, bigrams, trigrams, k):
    '''
    Prints out the k most frequent grams for each n-gram.
    Converts the input dictionaries into Counters for the easy method to do this. 

    :param unigrams: Dictionary of the counts for each 1-gram.
    :param bigrams: Dictionary of the counts for each 2-gram.
    :param trigrams: Dictionary of the counts for each 3-gram.
    :param k: The amount of grams to return for each dictionary.
    :return: Only prints out the results to show the user, no return. 
    '''

    unigram_counter = Counter(unigrams)
    bigram_counter = Counter(bigrams)
    trigram_counter = Counter(trigrams)
    print("Common n-grams")
    print(unigram_counter.most_common(k))
    print(bigram_counter.most_common(k))
    print(trigram_counter.most_common(k))
    print("\n")


def predict(text, unigrams, bigrams, trigrams):
    '''
    Function will take the different n-grams trained on and predict what the next word is after the text input.
    Prints out both the next word and probability for each of the n-grams.
    
    :param text: The text string to predict what comes next.
    :param unigrams: Dictionary of unigrams trained on.
    :param bigrams: Dictionary of bigrams trained on.
    :param trigrams: Dictionary of trigrams trained on.
    :return: Just prints out the result, does not return anything.
    '''
    
    num_words = len(unigrams)
    # These next few lines will determine the 1-gram language model.
    # Since it does not involve the text input, it will just determine the 3 most common words in the unigrams

    common_unigrams = Counter(unigrams).most_common(n=3)

    # Chance for a certain unigram is the number of times it has appeared (plus 1) divided by the total num of tokens
    print("1-gram Language Model Prediction")
    print(common_unigrams[0][0] + " " + str(((common_unigrams[0][1] + 1)/(sum(unigrams.values()) + num_words))))

    # From here, the 2-gram and 3-gram language models are done.
    # In case only one word in sentence, need to insert a beginning of sentence symbol (3-grams).
    text = word_tokenize(text)
    text.insert(0, "<s>")

    # Getting the previous 1/2 tokens
    # Case-folding since the training data was case-folded
    last_word = text[-1].casefold()
    last_two_words = [i.casefold() for i in text[-2:]]

    # Getting count of n-gram in the dictionary

    last_word_count = unigrams[last_word]
    last_two_words_count = bigrams[" ".join(last_two_words)]

    # Counter will help for getting the highest counts
    potential_bigrams = Counter()
    potential_trigrams = Counter()
    for bigram in bigrams:
        # Make sure the bigram starts with the right token (the end of the text input)
        if bigram.split()[0] == last_word:
            # The next potential token will be added to the bigram counter
            # with value of the prob. of it occurring given the previous token's count from the dictionary
            potential_bigrams[bigram.split()[1]] = ((bigrams[bigram]+1)/(last_word_count+num_words))

    for trigram in trigrams:
        # Make sure the trigram starts with the right bigram (last two tokens)
        if trigram.split()[0] == last_two_words[0] and trigram.split()[1] == last_two_words[1]:
            # The next potential token will be added to the trigram counter
            # with value of the prob. of it occurring given the previous bigram's count from the dictionary
            potential_trigrams[trigram.split()[2]] = ((trigrams[trigram]+1)/(last_two_words_count+num_words))

    # Want the three most likely next tokens printed with their probabilities

    common_bigrams = potential_bigrams.most_common(n=3)

    print("2-gram Language Model Prediction")
    print(common_bigrams[0][0] + " " + str(common_bigrams[0][1]))

    common_trigrams = potential_trigrams.most_common(n=3)

    print("3-gram Language Model Prediction")
    print(common_trigrams[0][0] + " " + str(common_trigrams[0][1]))
