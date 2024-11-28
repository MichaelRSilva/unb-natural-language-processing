import csv
import sys
from sklearn.model_selection import train_test_split

csv.field_size_limit(sys.maxsize)

def read_csv(filename, text_index, label_index):
    _text = []
    _label = []
    filename = f'dataset/{filename}'
    with open(filename, mode ='r') as file:
        csv_file = csv.reader(x.replace('\0', '') for x in file)
        for lines in csv_file:
            _text.append(lines[text_index])
            _label.append(lines[label_index])
    return _text, _label


def split_data(data, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    :param data: Pandas DataFrame containing the dataset.
    :param test_size: Proportion of the dataset to include in the test split (default 0.2).
    :param random_state: Random state for reproducibility (default 42).
    :return: X_train, X_test, y_train, y_test
    """
    X = data['text']
    y = data['label']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
#