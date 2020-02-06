import numpy
import sys

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from os import path, listdir

# Class to read in and manipulate text data
class DataInterpreter:
    def __init__(self, datapath="./data/", training_frac = 0.5, bad_chars=".!?\"“”", line_skip = 2):
        self.datapath = datapath  # Path to data files
        self.bad_chars = bad_chars  # Characters to strip so we only keep words
        # Used later to skip the chapter name and chapter number that are at the start of each data file
        self.line_skip = line_skip
        self.training_frac = training_frac  # Fraction of data to set a;side for training

        # Retrieve a list of .txt files in the datapath directory
        self.data_file_list = self.find_data_files()
        # Split the data files into training and testing data, these will be handled differently
        self.training_files = self.data_file_list[:int(len(self.data_file_list)*self.training_frac)]
        self.testing_files = self.data_file_list[int(len(self.data_file_list)*self.training_frac):]

        # Initialize the tokenizer before we use it in textfiles_to_sequences
        self.tokenizer = Tokenizer()
        # Convert our text/word data to sequences of integers and obtain the length of the longest sequence
        self.max_length, self.sequences = self.textfiles_to_padded_sequences(self.data_file_list)


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

    # Function to raw text data to sequences of integers using the Tokenizer in keras.preprocessing.text
    def textfiles_to_padded_sequences(self, data_file_list, seq_sep = "\n", word_sep = " "):
        # Open files
        for cnt, file in enumerate(data_file_list):
            if cnt > 0: continue ### Remove this later
            with open(file, "r", encoding="utf8") as datafile:
                txtdata = datafile.read()
                # Remove punctuation
                for i in range(len(self.bad_chars)):
                    txtdata = txtdata.replace(self.bad_chars[i],"")
        # Fit the tokenizer on the data to create a word index
        self.tokenizer.fit_on_texts([txtdata])
        # Loop over lines of text in our data and convert words to integers
        sequences = []
        for cnt, line in enumerate(txtdata.split(seq_sep)):
            if cnt < self.line_skip:  # Skipping the first N lines (due to chapter names/numbers..)
                continue
            converted_line = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(converted_line)):
                sequences.append(converted_line[: i + 1])
        # Max length of a text sequence, used later to set the size of the input layer in our NN
        max_length = numpy.max([len(sequence) for sequence in sequences])
        # Pad sequences so they are all consistent in length
        sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
        return max_length, sequences
