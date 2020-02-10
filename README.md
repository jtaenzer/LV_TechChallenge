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

#### Training:

```
python training.py
```

This will create a single LSTM layer sequential NN and train it on sequences of words from the training data.
The vocabulary from the training data will be saved in a json so it can be re-used for testing.
After training the NN will be saved in a .h5 file.

*WARNING* The training can be quite time consuming! I used a random sampling of 51 files from the first half of the data 
provided to produce the model file and tokenizer that are included in this repo.

#### Thoughts and caveats on the training:

General thoughts:

My approach was based on the idea that if I trained a NN on sequences of word tokens from the provided data set, given 
an arbitrary sequence of word tokens of length d the NN could return the probabilities for what the next word token 
should be. Testing with a large number of arbitrary sequences of length d and averaging the 
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
used a max sequence length of 15. Sequences longer than 15 words are still used but broken up into multiple sequences.

- I split the provided data into training and testing data simply by making an alphabetically ordered list and dividing 
it in two. This may introduce some bias since the vocabulary in the first half and second half of the data may differ,
and a better approach would have been to sample randomly. I never got around to implementing that, however.

#### Testing:

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

#### Accuracy assessment

Produce plots comparing P(W|d) from the above testing script with P(W|d) measured directly from the testing data:

```
python accuracy_assessment.py -n <num_words> -d <max_distance> -t <tests>
```

The same simplifications made to the text during training are also made to the testing data before measuring P(W|d).

Plots produced with -w 20 -d 15 -t 10000 can be found in the plots directory and are also included below.

Aside from the plots, my accuracy assessment won't be quantitative, because it can be summed up fairly
quickly: This approach did not produce accurate results! There are some interesting trends

- The model over-predicts the frequency of the most common words,
before turning over and under-predicting the frequency of less common words. That happens around the eighth most common
word ('yupela') and is quite distinct.

- The decay of P(W|d) with increasing d, measured from the test data is quite clear for the more common words 
(e.g. see the plots for 'i', 'na', 'bilong', 'ol'). Starting from around the 7th most common word ('man') we start to 
see fluctuations from the expected behavior that I would assume are caused by lack of statistics in the test data.

![i](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/i.png "i.png")

![na](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/na.png "na.png")

![bilong](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/bilong.png "bilong.png")

![ol](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/ol.png "ol.png")

![long](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/long.png "long.png")

![em](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/em.png "em.png")

![man](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/man.png "man.png")

![yupela](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/yupela.png "yupela.png")

![dispela](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/dispela.png "dispela.png")

![yu](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/yu.png "yu.png")

![olsem](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/olsem.png "olsem.png")

![mekim](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/mekim.png "mekim.png")

![go](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/go.png "go.png")

![bai](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/bai.png "bai.png")

![bikpela](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/bikpela.png "bikpela.png")

![tok](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/tok.png "tok.png")

![lain](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/lain.png "lain.png")

![bin](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/bin.png "bin.png")

![stap](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/stap.png "stap.png")

![tokim](https://github.com/jtaenzer/LV_TechChallenge/blob/master/plots/tokim.png "tokim.png")

## Credits

Not being familiar with neural language modeling,  I would have been lost without this tutorial:

https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/

My solution is built on the above!