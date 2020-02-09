# Imports of built-in libraries
import sys
import getopt
from os import environ
from json import load
from random import choice
# This will stop tensorflow from spamming unnecessary error messages about its GPU implementation
# Needs to set before we import anything from keras/tensorflow
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Import model class
from nn_model import NNModel
# Import keras libraries
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences


# Load the tokenizer used during training from a json
def load_tokenizer():
    with open('./tokenizer_51_file_training.json') as jsonf:
        data = load(jsonf)
        tokenizer = tokenizer_from_json(data)
    return tokenizer


# Load the model used during training
def load_model():
    model = NNModel()
    model.load_model("./model_51_file_training.h5")
    print(model.model.summary())
    return model


def print_help():
    print("Usage:")
    print("python testing.py -w <word> -d <distance> -t <tests>")
    print("Where")
    print("<word> should by a word in the word list used during training")
    print("<distance> is the distance between two instances of <word>")
    print("<tests> is the number of tests to perform with random sequences of words not containing <word>")
    print("To obtain the word list, try:")
    print("python testing.py -l")


def word_list():
    print(load_tokenizer().word_index.keys())


def main():
    # Retrieve arguments, print help() if that fails
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hlw:d:t:", ["help", "list", "word=", "distance=", "tests="])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    test_word = None
    distance = None
    num_tests = None
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()
            sys.exit()
        elif opt in ("-l", "--list"):
            word_list()
            sys.exit()
        elif opt in ("-w", "--word"):
            test_word = arg
        elif opt in ("-d", "--distance"):
            try:
                distance = int(arg)
            except ValueError:
                print("--distance %s couldn't be converted to an int. \n" % arg)
                print_help()
                sys.exit(2)
        elif opt in ("-t", "--tests"):
            try:
                num_tests = int(arg)
            except ValueError:
                print("--tests %s couldn't be converted to an int. \n" % arg)
                print_help()
                sys.exit(2)

    # Check that the script recieved all necessary arguments, print help if not
    if None not in (test_word, distance, num_tests):
        pass
    else:
        print_help()
        sys.exit(2)

    # Start by opening the tokenizer used during training
    # Will want to use the word_index and word_counts
    tokenizer = load_tokenizer()
    # Get the index of our test word in the tokenizer
    test_word_index = tokenizer.word_index[test_word]
    print(len(tokenizer.word_index))
    # Get the num_words setting for this tokenizer, used to determine which words were used during training
    num_words = tokenizer.num_words
    # Sort the words in the tokenizer by their frequency and keep only the first num_words words
    sorted_words = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)[:num_words]
    # Keep only the words, not the word counts, and remove the test word
    sorted_words = [x[0] for x in sorted_words]
    # Try to remove the test_word from sorted words
    try:
        sorted_words.remove(test_word)
    except ValueError:
        print("--word %s wasn't found in word list, printing word list and exiting." % test_word)
        word_list()
        sys.exit(2)

    # Load the model used during training
    model = load_model()

    probability_sum = 0
    for _ in range(num_tests):
        # Create a random sequence of words of length --distance that doesn't include
        random_words = ""
        for _ in range(distance):
            random_words += " " + choice(sorted_words)
        sequence = tokenizer.texts_to_sequences([random_words])[0]
        sequence = pad_sequences([sequence], maxlen=model.model.input_shape[1], padding="pre")
        probabilities = model.get_probability(sequence)[0]
        # We subtract 1 from test_word_index here since tokenizer.word_index starts from 1 but
        # model.get_probability returns a numpy array indexed from 0
        probability_sum += probabilities[test_word_index-1]

    print("Probability of encountering the word %s after a sequence of %d words is %f"
          % (test_word, distance, probability_sum/num_tests))


if __name__ == "__main__":
    main()
