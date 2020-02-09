from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

class NNModel():
    def __init__(self):
        self.model = None

    # Prepare the NN model
    def prepare_model(self, input_size, output_size, projection_size=32, hidden_layer_size=75):
        self.model = Sequential()
        self.model.add(Embedding(output_size, projection_size, input_length=input_size))
        self.model.add(LSTM(hidden_layer_size))
        self.model.add(Dense(output_size, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        print(self.model.summary())

    # Fit the model to the data
    def fit_model(self, input_data, output, epochs=500, verbosity=2):
        self.model.fit(input_data, output, epochs=epochs, verbose=verbosity)

    # Save the model to a .h5 file so it can be retrieved for later use
    def save_model(self, path="./model.h5"):
        self.model.save(path)

    # Load a model from a .h5 file
    def load_model(self, path="./model.h5"):
        self.model = load_model(path)

    # Given a sequence of integer -> word associations, generate a new integer
    def generate_word(self, sequence):
        return self.model.predict_classes(sequence, verbose=0)

    # Given a seed word or words, generate a sentence that is length words long
    def generate_sentence(self, seed_word, length, tokenizer):
        input_text = seed_word
        for _ in range(length):
            sequence = tokenizer.texts_to_sequences([input_text])[0]
            # Pad the sequence up to the size of the input layer, given by model.input_shape[1]
            sequence = pad_sequences([sequence], maxlen=self.model.input_shape[1], padding="pre")
            out_int = self.generate_word(sequence)
            out_word = ""
            for word, index in tokenizer.word_index.items():
                if index == out_int:
                    out_word = word
                    break
            input_text += " " + out_word
        return(input_text)

    # Given a sequence of integer -> word associations, return the probabilities of what the next word will be
    def get_probability(self, seed_sequence):
        return self.model.predict_proba(seed_sequence)