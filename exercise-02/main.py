# import os
# import re
# import json
# import nltk
#
# from nltk.corpus import stopwords
# from unidecode import unidecode
# from collections import defaultdict, Counter
#
# nltk.download('stopwords')
# sw = stopwords.words('portuguese')
#
# DATA_PATH = './data/'
#
# def normalize_text(text):
#
#     # lowercase everything and remove line breaks
#     normalized_text = text.lower().replace("\n", ' ')
#
#     # remove accents
#     normalized_text = unidecode(normalized_text)
#
#     # remove stopwords
#     normalized_text = ' '.join([k for k in normalized_text.split(" ") if k not in sw or len(k) > 2])
#
#     # remove all non-alpha-numeric symbols to leave only words and numbers
#     normalized_text = re.sub('[^a-z]', ' ', normalized_text)
#
#     # collapse multiple spaces into a single space
#     normalized_text = re.sub(' +', ' ', normalized_text)
#
#     return normalized_text
#
# def get_text():
#     train_text_list = []
#     test_text_list = []
#     all_files = [file for file in os.listdir(DATA_PATH) if file.endswith('.json')]
#     count_train = int(len(all_files)*0.8)
#     file_index = 0
#     for file_name in all_files:
#         with open(DATA_PATH + file_name) as json_file:
#             data = json.load(json_file)
#             file_index += 1
#             normalized_text = normalize_text(data["text"])
#             list_text = normalized_text.split(" ")
#
#             if file_index >= count_train:
#                 test_text_list.append(list_text)
#             else:
#                 train_text_list.append(list_text)
#
#
#     return train_text_list, test_text_list
#
#
# def build_bigrams_model(text):
#     """
#     Build a bigram model from the input text.
#     :param text: List of sentences, where each sentence is a list of words.
#     :return: Dictionary of bigram probabilities.
#     """
#     bigrams_counts = defaultdict(Counter)
#     unigram_counts = Counter()
#
#     # Count bigrams and unigrams
#     for sentence in text:
#         for i in range(len(sentence) - 1):
#             unigram_counts[sentence[i]] += 1
#             bigrams_counts[sentence[i]][sentence[i+1]] += 1
#         unigram_counts[sentence[-1]] += 1  # Count the last word in the sentence
#
#     # Calculate bigrams probabilities
#     bigrams_probs = defaultdict(dict)
#     for word, next_words in bigrams_counts.items():
#         total_count = unigram_counts[word]
#         for next_word, count in next_words.items():
#             bigrams_probs[word][next_word] = count / total_count
#
#     return bigrams_probs
#
# def next_word_probability(word, bigrams_probs):
#     """
#     Find the next word based on the highest bigram probability.
#     :param word: The current word.
#     :param bigram_probs: Dictionary of bigram probabilities.
#     :return: The next word with the highest probability.
#     """
#     if word not in bigrams_probs:
#         return None  # No next word found
#     next_word = max(bigrams_probs[word], key=bigrams_probs[word].get)
#     return next_word
#
# def calculate_perplexity(sentence, bigrams_probs):
#     """
#     Calculate the perplexity of a sentence using bigram probabilities.
#     :param sentence: List of words in the sentence.
#     :param bigrams_probs: Dictionary of bigram probabilities.
#     :return: Perplexity score.
#     """
#     perplexity = 1
#     N = len(sentence)
#     for i in range(N - 1):
#         word, next_word = sentence[i], sentence[i + 1]
#         probability = bigrams_probs.get(word, {}).get(next_word, 1e-6)  # Small probability for unseen bigrams
#         perplexity *= 1 / probability
#     perplexity = perplexity ** (1 / N)
#     return perplexity
#
#
# def generate_sentence(start_word, size, model):
#     words = []
#     w = start_word
#     for i in range(size):
#         w = next_word_probability(w, model)
#         words.append(w)
#     return (' '.join(words) + '. ').capitalize()
#
#
# if __name__ == '__main__':
#     # Example usage:
#     train_text, test_text = get_text()
#
#     # TODO
#     # criar senença igual o bigram.py
#     # calcular a perplexidade
#
#     # Build bigrams model
#     bigrams_model = build_bigrams_model(train_text)
#
#     # Test next word prediction
#     word = "brasileiro"
#     story = generate_sentence(word, 20, bigrams_model)
#
#     print(story)
#
#     # Calculate perplexity of a sentence
#     print(f"Perplexity of the sentence: {calculate_perplexity(test_text[10], bigrams_model)}")
#
#
# # https://datachild.net/machinelearning/bigram-language-model-python
# # https://github.com/thiagodepaulo/nlp/blob/main/aula_2/exercicio2.md