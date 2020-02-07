from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

class NNModel():
    def __init__(self):
        self.model = None

    def prepare_model(self, input_size, output_size, projection_size=32, num_hidden_neurons=75):
        self.model = Sequential()
        self.model.add(Embedding(output_size, projection_size, input_length=input_size))
        self.model.add(LSTM(num_hidden_neurons))
        self.model.add(Dense(output_size, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit_model(self, input_data, output, epochs=500, verbosity=2):
        self.model.fit(input_data, output, epochs=epochs, verbose=verbosity)

    def save_model(self, path="./model.h5"):
        self.model.save(path)

    def load_model(self, path="./model.h5"):
        self.model = load_model(path)