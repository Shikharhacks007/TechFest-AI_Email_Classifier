import pandas as pd
import os
import random


def draw_data(set):
    DATADIR = r'D:\archive'

    CATEGORIES = ['business', 'entertainment', 'food', 'graphics', 'historical', 'medical', 'politics',
                  'space', 'sport', 'technology']

    training_data_core = []

    corpus = []

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for txt in os.listdir(path):
            txt_array = open(os.path.join(path, txt), encoding="utf8")
            txt_array = txt_array.read()
            training_data_core.append([txt_array.replace("\n", ""), class_num])

    if set in CATEGORIES:
        n = 0
        corpus = training_data_core[CATEGORIES.index(set) * 100:(CATEGORIES.index(set) + 1) * 100]
        for rand_data in random.choices(training_data_core[0:CATEGORIES.index(set) * 100] +
                                        training_data_core[(CATEGORIES.index(set) + 1) * 100:-1], k=100):
            corpus += [rand_data]

        for i in corpus:
            if i[1] == CATEGORIES.index(set):
                i[1] = True
            else:
                i[1] = False
        random.shuffle(corpus)
    Corpus = pd.DataFrame(corpus).rename(columns={0: "text", 1: "label"})
    return Corpus

# print(draw_data("food"))
