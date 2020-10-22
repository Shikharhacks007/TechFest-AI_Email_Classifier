'''
1) Open vs code, and in the terminal type:
activate tf, OR -
2)Run conda prompt and activate the tensorflow environment
conda activate tf
Then, open VS Code and Check if the VS Code terminal is in the (tf) environment
'''

# Importing csv, numpy and tensorflow
import csv

import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional

# Importing and setting up nltk
import nltk
##nltk.download("stopwords")

from nltk.corpus import stopwords
STOPWORDS = set( stopwords.words('english') )

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

#with open("./Multi Class Text Classification with LTSM/bbc-text.csv", "r") as csvfile:
with open( "./bbc-text.csv", "r" ) as csvfile:
    reader = csv.reader( csvfile, delimiter=',' )
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
print( "Number of labels:", len(labels) )      #2225
print( "Number of articles:", len(articles) )  #2225

# Creating the Training and Validation Set
train_size = int( len(articles) * training_portion )

train_articles = articles[ : train_size ]
train_labels = labels[ : train_size ]

validation_articles = articles[ train_size : ]
validation_labels = labels[ train_size : ]

print( "Train Size:", train_size )                           #1780
print( f"train_articles: {len(train_articles)}" )            #1780
print( f"train_labels: {len(train_labels)}" )                #1780
print( f"validation_articles: {len(validation_articles)}" )  #445
print( f"validation_labels: {len(validation_labels)}" )      #445

# Tokenization
# The method fit_on_texts creates a vocabulary index based on word frequency

# Example: tokenizer.fit_on_texts( ["The cat sat on the mat."] ) 
# The corresponding dictionary would be: 
# tokenizer.word_index = {'<OOV>': 1, 'cat': 3, 'mat': 6, 'on': 5, 'sat': 4, 'the': 2 }
# If the word is not found in the dictionary, it gets added to <OOV>

tokenizer = Tokenizer( num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts( train_articles )
word_index = tokenizer.word_index

# ML works well with numbers, so the method texts_to_sequences transforms each text in texts to a sequence of ints.
# It basically converts each word into its corresponding integer value from the dictionary - tokenizer.word_index

# Example: tokenizer.texts_to_sequences( ["the cat sat on my table"] )  -> [ [2,3,4,5,1,1] ]
train_sequences = tokenizer.texts_to_sequences( train_articles )
validation_sequences = tokenizer.texts_to_sequences( validation_articles )

# Now, the sequences have different sizes, most sentences in English do not have the same size. Duh
# By padding and truncating, this issue will be solved.
# This is where the hyperparameters are being utilized.

train_padded = pad_sequences( train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type )
validation_padded = pad_sequences( validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type )
# When the padding_type and truncating_type have been set to 'post', zeroes will be added to fill in the gaps
# towards the end.
# When the padding_type and truncating_type have been set to 'pre', we get the zeroes at the beginning.

# Checking out the labels:
print( "Labels:", set(labels) )         #{'entertainment', 'business', 'sport', 'politics', 'tech'}

# Tokenization of Labels:
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts( labels )

training_label_seq = np.array( label_tokenizer.texts_to_sequences( train_labels ) )
validation_label_seq = np.array( label_tokenizer.texts_to_sequences( validation_labels ) )
print( "Word Index of Label Tokenizer: ", label_tokenizer.word_index )

# Creating the Sequential Model:
model = Sequential( )
# Hyperparameters being used here as well

# Adding layers to the model:
model.add( Embedding( vocab_size, embedding_dim ) )
'''
The model begins with an embedding layer which turns the input integer indices
into the corresponding word vectors. Word embedding is a way to represent a
word as a vector. It allows the values of the vector's elements to e trained.
After training, words with similar meanings often have similar vectors.
'''

model.add( Dropout(0.5) )
'''
Dropout is a regularization method that approximates training a large number of neural networks 
with different architectures in parallel. During training, some number of layer outputs are randomly 
ignored or “dropped out.” This has the effect of making the layer look-like and be treated-like a 
layer with a different number of nodes and connectivity to the prior layer.
This prevents overfitting of the model.
'''

model.add( Bidirectional( LSTM( embedding_dim ) ) )
'''
The bidirectional layer propagates the input forward and backward through the LSTM layer and then
concatenates the output. Helps the model learn long range dependencies.
'''

model.add( Dense( 6, activation='softmax' ) )
'''
This is the final layer, with softmax activation for the multi class classification.
'''

print( "Summary of the model:\n" )
model.summary()

# Compiling the model:
# Since we did not One Hot Encode the labels, we need to configure the training process with
# the sparse_categorical_crossentropy ( using the Adam optimizer )

opt = tf.keras.optimizers.Adam( lr=0.001, decay=1e-6 )

model.compile( loss      = 'sparse_categorical_crossentropy',
               optimizer = opt,
               metrics   = ['accuracy']  )


# Now, we are ready to train the model using the method fit()
num_epochs = 10
history = model.fit( train_padded, training_label_seq, epochs=num_epochs, 
                     validation_data=(validation_padded, validation_label_seq),
                     verbose = 2    )

print( "Model After Training:\n", history )




print("Done")

