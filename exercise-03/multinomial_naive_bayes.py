import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from util import split_data
from find_best_hyperparameters import find_best_hyperparameters


def multinomial_naive_bayes(data, num_classes, basename):
    """
    Implements Multinomial Naive Bayes with Grid Search for parameter tuning.
    Retrains the model on the full training data with the best parameters.
    Evaluates performance using F1 Score and Accuracy on a test dataset.

    :param data: Pandas DataFrame containing the dataset with 'text' and 'label' columns.
    :return: None (prints results).
    """

    print("Running ")

    # Split the dataset
    X_train, X_test, y_train, y_test = split_data(data)

    # Define a pipeline with TfidfVectorizer and Multinomial Naive Bayes
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])

    # Hyperparameter grid for GridSearchCV
    param_grid = {
        'vectorizer__min_df': [1, 2, 3],               # Minimum document frequency
        'vectorizer__max_df': [0.8, 0.9, 1.0],         # Maximum document frequency
        'vectorizer__ngram_range': [(1, 1), (1, 2)],   # N-gram ranges
        'classifier__alpha': [0.1, 0.5, 1.0, 5.0],     # Smoothing parameter
        'classifier__fit_prior': [True, False],        # Whether to learn class prior probabilities
    }

    # Perform Grid Search
    best_model, best_param = find_best_hyperparameters(param_grid, pipeline, X_train, y_train, num_classes)
    print(f"Best Parameters multinomial_naive_bayes and {basename}: {best_param}")
    df_best_param = pd.DataFrame.from_dict(best_param, orient="index")
    df_best_param.to_csv(f'multinomial_naive_bayes-{basename}', index=False)


    # Predict on test set
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)

    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")