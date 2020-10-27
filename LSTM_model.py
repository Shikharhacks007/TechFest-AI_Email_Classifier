import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


STOPWORDS = set(stopwords.words('english'))

# Setting up the hyperparameters
vocab_size = 5000    # Top number of common words to consider
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'   # OOV = Out of Vocabulary
training_portion = 0.8

# Populating the list and removing stopwords
articles = []
labels = []


# with open("./Multi Class Text Classification with LTSM/bbc-text.csv", "r") as csvfile:
with open( "./bbc-text.csv", "r" ) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)

    for row in reader:
        labels.append( row[0] )
        article = row[1]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace( token, ' ')
            article = article.replace( ' ', ' ')

        articles.append( article )

# Printing total number of labels and articles
print("Number of labels:", len(labels))
print("Number of articles:", len(articles))

# Creating the Training and Validation Set
train_size = int(len(articles) * training_portion)

train_articles = articles[0: train_size]
train_labels = labels[0: train_size]

validation_articles = articles[train_size:]
validation_labels = labels[train_size:]


print("train_size", train_size)
print(f"train_articles {len(train_articles)}")
print("train_labels", len(train_labels))
print("validation_articles", len(validation_articles))
print("validation_labels", len(validation_labels))


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index


train_sequences = tokenizer.texts_to_sequences(train_articles)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


# Tokenization
# The method fit_on_texts creates a vocabulary index based on word frequency
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index


# ML works well with numbers, so the method texts_to_sequences transforms each text in texts to a sequence of ints.
# It basically converts each word into its corresponding integer value from the dictionary - tokenizer.word_index
# Example: tokenizer.texts_to_sequences( ["the cat sat on my table"] )  -> [ [2,3,4,5,1,1] ]
train_sequences = tokenizer.texts_to_sequences(train_articles)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Tokenization of Labels:
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

# Creating the Sequential Model:
model = Sequential()
# Hyperparameters being used here as well

# Adding layers to the model:
# The model begins with an embedding layer which turns the input integer indices
# into the corresponding word vectors. Word embedding is a way to represent a
# word as a vector. It allows the values of the vector's elements to e trained.
# After training, words with similar meanings often have similar vectors.
model.add(Embedding(vocab_size, embedding_dim))

'''
Dropout is a regularization method that approximates training a large number of neural networks 
with different architectures in parallel. During training, some number of layer outputs are randomly 
ignored or “dropped out.” This has the effect of making the layer look-like and be treated-like a 
layer with a different number of nodes and connectivity to the prior layer.
This prevents overfitting of the model.
'''
model.add(Dropout(0.5)) ####

'''
The bidirectional layer propagates the input forward and backward through the LSTM layer and then
concatenates the output. Helps the model learn long range dependencies.
'''
model.add(Bidirectional(LSTM(embedding_dim)))

'''
This is the final layer, with softmax activation for the multi class classification.
'''
model.add(Dense(6, activation='softmax')) ####



# Compiling the model:
# Since we did not One Hot Encode the labels, we need to configure the training process with
# the sparse_categorical_crossentropy ( using the Adam optimizer )
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
    )

num_epochs = 15

# Now, we are ready to train the model using the method fit()v
history = model.fit(train_padded, training_label_seq, epochs=num_epochs,
                    validation_data=(validation_padded, validation_label_seq), verbose=2)


# to plot the graph of loss
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

text=['''China net cafe culture crackdown

Chinese authorities closed 12,575 net cafes in the closing months of 2004, the country's government said.

According to the official news agency most of the net cafes were closed down because they were operating illegally. Chinese net cafes operate under a set of strict guidelines and many of those most recently closed broke rules that limit how close they can be to schools. The move is the latest in a series of steps the Chinese government has taken to crack down on what it considers to be immoral net use.

The official Xinhua News Agency said the crackdown was carried out to create a "safer environment for young people in China". Rules introduced in 2002 demand that net cafes be at least 200 metres away from middle and elementary schools. The hours that children can use net cafes are also tightly regulated. China has long been worried that net cafes are an unhealthy influence on young people. The 12,575 cafes were shut in the three months from October to December. China also tries to dictate the types of computer games people can play to limit the amount of violence people are exposed to.

Net cafes are hugely popular in China because the relatively high cost of computer hardware means that few people have PCs in their homes. This is not the first time that the Chinese government has moved against net cafes that are not operating within its strict guidelines. All the 100,000 or so net cafes in the country are required to use software that controls what websites users can see. Logs of sites people visit are also kept. Laws on net cafe opening hours and who can use them were introduced in 2002 following a fire at one cafe that killed 25 people. During the crackdown following the blaze authorities moved to clean up net cafes and demanded that all of them get permits to operate. In August 2004 Chinese authorities shut down 700 websites and arrested 224 people in a crackdown on net porn. At the same time it introduced new controls to block overseas sex sites. The Reporters Without Borders group said in a report that Chinese government technologies for e-mail interception and net censorship are among the most highly developed in the world.
''']


def pred(text):
    seq = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(seq, maxlen=max_length)
    pred = model.predict(padded)
    labels = ['sport', 'bussiness', 'politics', 'tech', 'entertainment']  # orig
    print(pred)
    print(np.argmax(pred))
    print(labels[np.argmax(pred) - 1])

pred(text)

