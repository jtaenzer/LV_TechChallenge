# Imports of built-in libraries
import sys
import getopt
from os import environ
from json import load
from random import choice
from re import split
# This will stop tensorflow from spamming unnecessary error messages about its GPU implementation
# Needs to set before we import anything from keras/tensorflow
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Import data interpreter and model classes
from data_interpreter import DataInterpreter
from nn_model import NNModel
# Import keras libraries
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
# Import matplotlib
import matplotlib.pyplot as plt


def print_help():
    print("Usage:")
    print("python accuracy_assessment.py -n <num_words> -d <max_distance> -t <tests")
    print("Where")
    print("<num_words> is the number of words to assess from tokenizer word_index (sorted by most common)")
    print("<max_distance> is the largest distance to check between words and should be an integer number")
    print("<tests> is the number of tests to perform with random sequences of words")
    print("To obtain the word list, try:")
    print("python testing.py -l")
    print("To obtain the word list, try:")
    print("python testing.py -l")


def word_list():
    print(load_tokenizer().word_index.keys())


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


def main():
    # Retrieve arguments, print help() if that fails
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hln:d:t:", ["help", "list", "numwords=", "maxdistance=", "tests="])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    num_words = None
    min_d = 3
    max_d = None
    num_tests = None
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()
            sys.exit()
        elif opt in ("-l", "--list"):
            word_list()
            sys.exit()
        elif opt in ("-n", "--numwords"):
            try:
                num_words = int(arg)
            except ValueError:
                print("--numwords %s couldn't be converted to an int. \n" % arg)
                print_help()
                sys.exit(2)
        elif opt in ("-d", "--maxdistance"):
            try:
                max_d = int(arg) + 1  # Add one since we will use a range that ends at max_d
            except ValueError:
                print("--maxdistance %s couldn't be converted to an int. \n" % arg)
                print_help()
                sys.exit(2)
        elif opt in ("-t", "--tests"):
            try:
                num_tests = int(arg)
            except ValueError:
                print("--tests %s couldn't be converted to an int. \n" % arg)
                print_help()
                sys.exit(2)

    # Load the tokenizer and get a list of the words used for training
    tokenizer = load_tokenizer()
    sorted_word_counts = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)[:tokenizer.num_words]
    sorted_words = [x[0] for x in sorted_word_counts]

    data_interp = DataInterpreter()
    # Read the testing data
    txtdata = data_interp.read_text_files(data_interp.testing_files)
    # Make the same simplficiations to our testing data that we did with our training data
    txtdata = data_interp.simplify_text_data_with_tokenizer(txtdata, tokenizer)

    # Load the trained model
    model = load_model()

    # Loop over sorted words
    for word in sorted_word_counts[:num_words]:
        print("Assessing accuracy for word %s" % word[0])
        model_prob_list = []
        testdata_prob_list = []
        # Split the txtdata by the current word using regular expressions to find word boundaries
        txtdata_split = split(r"\b%s\b" % word[0], txtdata)
        # Convert the txtdata to integer sequences using the tokenizer -- makes it easier to measure the distance
        converted_txt = tokenizer.texts_to_sequences(txtdata_split)

        # Now loop over the requested distances to check
        for dist in range(min_d, max_d):
            # Loop over sequences made from our testing data and count how many have the right distance
            cnt = 0
            for sequence in converted_txt:
                if len(sequence) == dist:
                    cnt += 1
            testdata_prob_list.append(cnt/len(txtdata_split))

            # Perform num_test tests with the model generating random word sequences
            test_word_index = tokenizer.word_index[word[0]]
            probability_sum = 0
            for _ in range(num_tests):
                # Create a random sequence of words of length --distance that doesn't include
                random_words = ""
                for _ in range(dist):
                    random_words += " " + choice(sorted_words)
                sequence = tokenizer.texts_to_sequences([random_words])[0]
                sequence = pad_sequences([sequence], maxlen=model.model.input_shape[1], padding="pre")
                probabilities = model.get_probability(sequence)[0]
                # We subtract 1 from test_word_index here since tokenizer.word_index starts from 1 but
                # model.get_probability returns a numpy array indexed from 0
                probability_sum += probabilities[test_word_index - 1]
            model_prob_list.append(probability_sum/num_tests)

        # Create a distance list to use in plotting
        distances = [x for x in range(min_d, max_d)]
        # Plot the probability vs distance for the model and from the test data
        plt.plot(distances, model_prob_list, label='Model', color='darkblue', marker='.')
        plt.plot(distances, testdata_prob_list, label='Test data', color='green', marker='^')
        plt.legend()
        plt.xlim(min_d-1, max_d)
        plt.xlabel("distance (# words)")
        plt.ylabel("Probability")
        plt.title(word[0])
        plt.savefig("./plots/%s.png" % word[0])
        plt.close()

if __name__ == "__main__":
    main()