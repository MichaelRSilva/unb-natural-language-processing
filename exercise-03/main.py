import pandas as pd

from multinomial_naive_bayes import multinomial_naive_bayes
from svm import svm
from logistic_regression import logistic_regression

from util import read_csv

if __name__ == "__main__":

    # webkb-parsed.csv | Dmoz-Health.csv
    basename = "Dmoz-Health.csv"
    text, label = read_csv(basename,1, 2)
    text = text[1:]
    label = label[1:]

    # text, label = read_csv(basename,'text', 'class')
    num_classes = len(list(set(label)))

    print(label)

    data = pd.DataFrame({
        'text': text,
        'label': label
    })

    # Apply the function
    multinomial_naive_bayes(data, num_classes, basename)
