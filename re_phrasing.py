import numpy as np
import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection


def draw_data(DATADIR, Set, label=None):
    CATEGORIES = os.listdir(DATADIR)

    training_data_core = pd.DataFrame(columns=['Text', 'Label'])

    # Obtains all data from classified txt files

    if os.path.splitext(DATADIR)[1] == "":
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            for txt in os.listdir(path):
                txt_array = open(os.path.join(path, txt), encoding="utf8")
                txt_array = txt_array.read()
                training_data_core = training_data_core.append({'Text': str(txt_array), 'Label': str(category)},
                                                               ignore_index=1)
    elif os.path.splitext(DATADIR)[1] == ".csv" and label == True:
        try:
            corpus = pd.read_csv(DATADIR)
        except:
            print('Please Enter as such ["The Data/ body of emails", "labels of them"] in list')

    if Set in CATEGORIES:
        corpus = training_data_core.query("Label == Set")["Label"] == True
        corpus = corpus.append(training_data_core.query("Label != category").sample(n=len(corpus))["Label"] == False)
    else:
        return "The Given Set {} was not found in the given Categories {}!".format(Set, CATEGORIES)

    ########################################################################################################################
    # Post Processing Starts

    # Step - 1a : Remove blank rows if any.
    corpus['Text'].dropna(inplace=True)

    # Step - 1b : Change all the Text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    corpus['Text'] = [entry.lower() for entry in corpus['Text']]

    # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
    corpus['Text'] = [word_tokenize(entry) for entry in corpus['Text']]

    # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.

    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is
    # set to Noun
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    for index, entry in enumerate(corpus['Text']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        corpus.loc[index, 'text_final'] = str(Final_words)

    print(corpus['text_final'].head())

    # Step - 2: Split the model into Train and Test Data set
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpus['text_final'], corpus['label'],
                                                                        test_size=0.3)

    # Step - 3: Label encode the target variable  - This is done to transform Categorical data of string type in the
    # data set into numerical values
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)

    # Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in
    # document is in comaprison to the corpus
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(corpus['text_final'])

    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    return Train_X_Tfidf, Test_X_Tfidf
