from json import dumps
from os import environ
# This will stop tensorflow from spamming unnecessary error messages about its GPU implementation
# Needs to set before we import anything from keras/tensorflow
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from data_interpreter import DataInterpreter
from nn_model import NNModel
from keras.utils import to_categorical


# Number of files to use for training
n_files = 51

# Create the DataInterpreter
data_interp = DataInterpreter()
# Read data files as a string
txtdata = data_interp.read_text_files(data_interp.training_files, n_files, sample_data=True)
# Simplify text files by replacing similar words, ignore words that appear less often than min_freq
# Using n_files/2 as the min_freq is a rule of thumb I determined empirically to keep the training time reasonable
txtdata = data_interp.simplify_text_data(txtdata, min_freq=n_files/2)
# Set the number of words to keep based on the number of words that appear more often min_feq
vocab = data_interp.set_num_words(txtdata, min_freq=n_files/2)
vocab_size = len(vocab) + 1
# Convert the data to sequences of integers with some maximum length
max_length, sequences = data_interp.training_data_to_padded_sequences(txtdata, max_len=15, shuffle_data=True)
# Break up the sequences into input (sequence of n words) and output (single word to test against)
input_data, output = sequences[:, :-1], sequences[:, -1]
output = to_categorical(output, num_classes=vocab_size)

# Save the tokenizer for later use, in case we randomized the training data
# If the training data was randomized we will need to know the words and word_index later for testing
tokenizer_json = data_interp.tokenizer.to_json()
with open("./tokenizer_%s_file_training.json" % n_files, "w", encoding="utf-8") as jsonf:
    jsonf.write(dumps(tokenizer_json, ensure_ascii=False))

# Prepare the model
model = NNModel()
# Input layer should have max_length - 1 neurons, output layer should have one neuron per word token
# Hidden layer size determined by the 2/3*(input layer + output layer) rule of thumb
model.prepare_model(max_length - 1, vocab_size, hidden_layer_size=int((vocab_size + max_length - 1)*2/3))
# Fit on training data
model.fit_model(input_data, output)
# Save model, can be loaded later for testing without re-training
model.save_model("./model_%s_file_training.h5" % str(n_files))