# -*- coding: utf-8 -*

import numpy
import sys
import itertools
from random import shuffle, sample
from os import path, listdir
from re import sub

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk import edit_distance

# Class to read in and manipulate text data
class DataInterpreter:
    def __init__(self, datapath="./data/", training_frac=0.5, line_skip=2):
        self.datapath = datapath  # Path to directory of data files
        self.training_frac = training_frac  # Fraction of data to set a;side for training
        self.data_file_list = self.find_data_files() # Retrieve a list of .txt files in the datapath directory
        # Split the data files into training and testing data, these will be handled differently
        # Probably introducing a bias here by splitting down the middle, using random choices might be better
        self.training_files = self.data_file_list[:int(len(self.data_file_list)*self.training_frac)]
        self.testing_files = self.data_file_list[int(len(self.data_file_list)*self.training_frac):]
        # Initialize the tokenizer
        self.tokenizer = Tokenizer()

    # Function that finds all .txt files in the self.datapath directory
    def find_data_files(self):
        # Ensure that datapath actually exists
        if not path.exists(self.datapath):
            print("DataInterpreter.textfiles_to_sequences : %s directory couldn't be found, exiting." % self.datapath)
            sys.exit(2)
        # Add a trailing / to datapath if its not there
        if not self.datapath.endswith("/"):
            self.datapath += "/"
        return [self.datapath + datafile for datafile in listdir(self.datapath) if datafile.endswith(".txt")]

    # Set the number of words to keep in the tokenizer based the frequency with which words appear in the txt data
    # Returns vocab size as this is used to determine the size of the hidden layer and output layer
    def set_num_words(self, txtdata, min_freq=100):
        tmp_tokenizer = Tokenizer()
        tmp_tokenizer.fit_on_texts([txtdata])
        # Sort the tokenizer word counts by the frequency with which each word appears
        sorted_word_counts = sorted(tmp_tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
        # Filter out words that appear less often than min_freq
        filtered_word_counts = list(filter(lambda x: x[1] > min_freq, sorted_word_counts))
        # Length of the filtered list is the new vocabulary size, i.e. tokenizer.num_words
        vocab_size = len(filtered_word_counts)
        self.tokenizer.num_words = vocab_size
        return filtered_word_counts

    # Convert text data line by line into padded sequences of integers, which we will feed into the model
    # Most of our data contains one sentence per line, hence the default sequence seperate of a line break
    # In some cases that formatting doesn't hold, so for sentences greater than max_len we will try to split by alt_seps
    def training_data_to_padded_sequences(self, txtdata, seq_sep="\n", max_len=50, shuffle_data=True):
        # Fit the tokenizer on the data to create a word index
        self.tokenizer.fit_on_texts([txtdata])
        # Split the text data to chunks using seq_sep so we can loop over the chunks
        txtdata_split = txtdata.split(seq_sep)
        # By default, shuffle the chunks
        if shuffle_data:
            shuffle(txtdata_split)
        # Loop over chunks of data and convert to sequences of integers
        sequences = []
        for chunk in txtdata_split:
            # Convert the chunk of text data to a sequence of integers
            converted_chunks = self.tokenizer.texts_to_sequences([chunk])[0]
            # If converted_chunks contains more words than max_len, split it up into chunks of length max_len
            if len(converted_chunks) > max_len:
                converted_chunks = [converted_chunks[i: i+max_len] for i in range(0, len(converted_chunks), max_len)]
            else:
                converted_chunks = [converted_chunks]  # Has to be a list so we can loop over it below
            for converted_chunk in converted_chunks:
                # Skip any empty converted_lines, could be left over from splitting
                if not converted_chunk:
                    continue
                # Create multiple sequences from each chunk building them up word by word
                for i in range(1, len(converted_chunk)):
                    sequences.append(converted_chunk[: i + 1])
        # Max length of text sequences, used for padding and later to set the size of the input layer in our NN
        # Should usually be equal to max_len argument, but could be smaller
        max_length = numpy.max([len(sequence) for sequence in sequences])
        # Pad sequences so they are all consistent in length
        sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
        return max_length, sequences

    # Similar function to the above training_data_to_padded_sequences
    # For the testing data we won't create multiple subsequnces from each sequence
    # We will also simply drop sequences exceeding the max_len for simplicity
    def test_data_to_padded_sequences(self, txtdata, seq_sep, max_len=50):
        # Split by seq_sep string, which in general will probably be a word, and loop over chunks
        sequences = []
        for chunk in txtdata.split(seq_sep):
            converted_chunk = self.tokenizer.texts_to_sequences([chunk])[0]
            # Drop sequences with length greater than max_len
            if len(converted_chunk) > max_len:
                continue
            sequences.append(pad_sequences([converted_chunk], maxlen=max_len-1, padding='pre'))
        # Pad sequences up to max_len, which should be equal to the vocabulary size / size of the input layer
        #sequences = pad_sequences(sequences, maxlen=max_len-1, padding='pre')
        return sequences

    # Read in text files from a list of paths, return a string of text data
    @staticmethod
    def read_text_files(input_file_list, n_files_to_read=-1, sample_data=False):
        # By default just use the entire dataset
        data_file_list = input_file_list
        # If we want a subset of the data and sample_data is true, select a random sample from the data
        if n_files_to_read > 0 and sample_data:
            data_file_list = sample(input_file_list, n_files_to_read)
        # Otherwise, just take the first n files
        elif n_files_to_read > 0 and not sample_data:
            data_file_list = input_file_list[:n_files_to_read]
        txtdata = ""
        # Loop over file list
        for file in data_file_list:
            # Open file and append the text to txtdata
            with open(file, "r", encoding="utf8") as datafile:
                tmpdata = datafile.read()
                # Remove utf8 curly quotes, tokenizer removes most punctuation but not these
                tmpdata = tmpdata.replace(u"\u201c", "")
                tmpdata = tmpdata.replace(u"\u201d", "")
                tmpdata = tmpdata.replace(u"\u2018", "")
                tmpdata = tmpdata.replace(u"\u2019", "")
                txtdata += tmpdata
        return txtdata

    # Simplify the text data by identifying similar pairs words and keeping only one of the pair
    # Similarity of words determined using the Levenstein's distance -- from the nltk library's edit_distance
    # This can be incredibly computation expensive for a large amount of text data so we will restrict it to common words
    @staticmethod
    def simplify_text_data(txtdata, min_dist=2, min_freq=100):
        tmp_tokenizer = Tokenizer()
        tmp_tokenizer.fit_on_texts([txtdata]) # Fit the textdata to get a word index
        # Sort the tokenizer word counts by the frequency with which each word appears
        sorted_word_counts = sorted(tmp_tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
        # Filter out words that appear less often than min_freq
        filtered_word_counts = list(filter(lambda x: x[1] > min_freq, sorted_word_counts))
        # Length of the filtered list is the new vocabulary size, i.e. tokenizer.num_words
        # Use itertools to get all possible unique pair-wise combinations of words in the filtered word_index
        pairs = list(itertools.combinations([filtered_word_counts[i][0] for i in range(len(filtered_word_counts))], 2))
        # Loop over pairs and calculate the distance, don't consider words that have already been replaced
        replaced_words = []
        for pair in pairs:
            dist = edit_distance(pair[0], pair[1])
            # If the distance is less than the min_dist and we haven't already replaced one of the words, replace
            if dist < min_dist and not (pair[0] in replaced_words or pair[1] in replaced_words):
                # Use regular expressions to identify word boundaries and do the string replacement
                txtdata = sub(r"\b%s\b" % pair[1], pair[0], txtdata)
                replaced_words.append(pair[1])
        return txtdata

    # Similar to the above simplify_text_data method, but we pass an existing tokenizer (one used for training)
    @staticmethod
    def simplify_text_data_with_tokenizer(txtdata, tokenizer, min_dist=2):
        # Determine the number of words to use for simplification from the tokenizer
        num_words = tokenizer.num_words
        if not num_words:  # If num_words wasn't set for the tokenizer, use all words
            num_words = len(tokenizer.word_index.items())
        # Sort the tokenizer word counts by the frequency with which each word appears
        # If num_words was set, our training will only have considered the first num_words most frequent words
        # So we sort the tokenizer by word counts and keep only the first num_words elements
        sorted_word_counts = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)[:num_words]
        # Use itertools to get all possible unique pair-wise combinations of words in sorted_word_counts
        pairs = list(itertools.combinations([sorted_word_counts[i][0] for i in range(len(sorted_word_counts))], 2))
        # Loop over pairs and calculate the distance, don't consider words that have already been replaced
        replaced_words = []
        for pair in pairs:
            dist = edit_distance(pair[0], pair[1])
            # If the distance is less than the min_dist and we haven't already replaced one of the words, replace
            if dist < min_dist and not (pair[0] in replaced_words or pair[1] in replaced_words):
                # Use regular expressions to identify word boundaries and do the string replacement
                from re import sub
                txtdata = sub(r"\b%s\b" % pair[1], pair[0], txtdata)
                replaced_words.append(pair[1])
        return txtdata