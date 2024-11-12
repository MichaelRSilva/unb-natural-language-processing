import os
import re
import json
import nltk

from nltk.corpus import stopwords
from unidecode import unidecode

nltk.download('stopwords')
sw = stopwords.words('portuguese')


def normalize_text(text):

    # lowercase everything and remove line breaks
    normalized_text = text.lower().replace("\n", ' ')

    # remove accents
    normalized_text = unidecode(normalized_text)

    # remove stopwords
    normalized_text = ' '.join([k for k in normalized_text.split(" ") if k not in sw or len(k) > 2])

    # remove all non-alpha-numeric symbols to leave only words and numbers
    normalized_text = re.sub('[^a-z]', ' ', normalized_text)

    # collapse multiple spaces into a single space
    normalized_text = re.sub(' +', ' ', normalized_text)

    return normalized_text

def get_text():
    train_text_list = []
    test_text_list = []
    data_path = './data/'
    all_files = [file for file in os.listdir(data_path) if file.endswith('.json')]
    count_train = int(len(all_files)*0.8)
    file_index = 0
    for file_name in all_files:
        with open(data_path + file_name) as json_file:
            data = json.load(json_file)
            file_index += 1
            normalized_text = normalize_text(data["text"])
            list_text = normalized_text.split(" ")

            if file_index >= count_train:
                test_text_list.append(list_text)
            else:
                train_text_list.append(list_text)


    return train_text_list, test_text_list
