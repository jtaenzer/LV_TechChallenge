# LV_TechChallenge

## Interpreter and dependencies
Environment: Windows 10

IDE: PyCharm CE

Interpreter used for testing: python 3.7.6

Dependencies: tensorflow, numpy, keras, keras-preprocessing, nltk, matplotlib

```
python -m pip install tensorflow --upgrade
python -m pip install keras --upgrade
python -m pip install keras-preprocessing --upgrade
python -m pip install numpy --upgrade
python -m pip install nltk --upgrade
python -m pip install matplotlib --upgrade
```

## Usage

The default setup *should* work out of the box if all dependencies are present.

Training:

```
python training.py
```

This will create a single LSTM layer sequential NN and train it on sentences from the training data.
The vocabulary from the training data will be saved in a json so it can be re-used for testing.
After training the NN will be saved in a .h5 file.

*WARNING* The training can be quite time consuming! I used a random sampling of 51 files from the first half of the data 
provided to produce the model file and tokenizer that are included in this repo.

### Thoughts and caveats on the training:

General thoughts:

My approach was based on the idea that if I trained a NN on sequences of words tokens from the provided data set, given 
an arbitrary sequence of word tokens of length d the NN could return the probabilities for what the next word token 
should be based on the training data. Testing with a large number of arbitrary sequences of length d and averaging the 
probabilities would than give me an approximation of P(W|d). In retrospect this is a flawed approach, as is made quite 
obvious by the accuracy assessment!

Caveats:

- Training the NN scales with the number of files not just because there is more data to train on but also because the 
size of the vocabulary increases. I made two simplifications to mitigate the issue of the vocbulary increasing:
1. Only considered words that appeared frequently in the text. This was aimed ignoring, for example, proper names that might
only be present in some small subset of the data.
2. Measured the [Levenstein's distance](https://www.nltk.org/_modules/nltk/metrics/distance.html) between all pairs of 
words in the vocabulary using the nltk library's edit_distance. For pairs with distance < 2, I replaced all instances of
of the less frequently appearing word with the more frequently appearing word.

- Even with the simplifications above, training on a random sampling of 50 files from the provided data still takes about 
1.5 hours. I've provided an already trained NN and the vocabulary used in the training so the testing and accuracy
assessment scripts can be used without retraining.

- A further simplification was implementing a maximum sequence length. Initially I was just giving the NN lines from the
text data, but discovered that some (rare) lines had lengths in excess of 150 words! Further investigation revealed that 
although for the most part 1 line = 1 sentence in the data, there are some exceptions to that rule where multiple
sentences appear on the same line. There are also some very long sentences. To avoid excessive padding of the input, I
used a max sequence length of 25. Sequences longer than 25 words are still used but broken up into multiple sequences.

- I split the provided data into training and testing data simply by making an alphabetically ordered list and dividing 
it in two. This may introduce some bias since the vocabulary in the first half and second half of the dat may differ,
and a better approach would have been to sample randomly. I never got around to implementing that, however.

Testing:

First get the word list:

```
python testing.py -l
```

And then run for any word from the word list:

```
python testing.py -w <WORD> -d <DISTANCE> -t <# OF TESTS>
```

This will generate -t random sequences of length -d from the word list, retrieve the
probability that the next word is -w for each sequence, and then average the sum of the probabilities by -t.

Major caveat:

The NN appears to be broken for the most common word, 'i', and always predicts a vanishingly small probability that
it will be the next word. Sadly I was never able to understand why this was is the case.

## Accuracy assessment

![i](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/i.png "i.png")


![na](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/i.png "na.png")


## Credits

Not being familiar with neural language modeling,  I would have been lost without this tutorial:

https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/

My solution is built on the above!