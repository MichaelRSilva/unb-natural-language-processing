from collections import defaultdict, Counter


def build_bigrams_model(text):
    """
    Build a bigrams model from the input text.
    :param text: List of sentences, where each sentence is a list of words.
    :return: Dictionary of bigrams probabilities.
    """
    bigrams_counts = defaultdict(Counter)
    unigram_counts = Counter()

    # Count bigrams
    for sentence in text:
        for i in range(len(sentence) - 1):
            unigram_counts[sentence[i]] += 1
            bigrams_counts[sentence[i]][sentence[i+1]] += 1
        unigram_counts[sentence[-1]] += 1  # Count the last word in the sentence

    # Calculate bigrams probabilities
    bigrams_probs = defaultdict(dict)
    for word, next_words in bigrams_counts.items():
        total_count = unigram_counts[word]
        for next_word, count in next_words.items():
            bigrams_probs[word][next_word] = count / total_count

    return bigrams_probs


def next_word_probability(word, bigrams_probs):
    """
    Find the next word based on the highest bigrams probability.
    :param word: The current word.
    :param bigrams_probs: Dictionary of bigrams probabilities.
    :return: The next word with the highest probability.
    """
    if word not in bigrams_probs:
        return None  # No next word found
    next_word = max(bigrams_probs[word], key=bigrams_probs[word].get)
    return next_word


def calculate_perplexity(sentence, bigrams_probs):
    """
    Calculate the perplexity of a sentence using bigrams probabilities.
    :param sentence: List of words in the sentence.
    :param bigrams_probs: Dictionary of bigrams probabilities.
    :return: Perplexity score.
    """
    perplexity = 1
    n = len(sentence)
    for i in range(n - 1):
        word, next_word = sentence[i], sentence[i + 1]
        probability = bigrams_probs.get(word, {}).get(next_word, 1e-6)  # Small probability for unseen bigrams
        perplexity *= 1 / probability
    perplexity = perplexity ** (1 / n)
    return perplexity


def generate_sentence(start_word, size, model):
    words = []
    w = start_word
    for i in range(size):
        w = next_word_probability(w, model)
        words.append(w)
    return (' '.join(words) + '. ').capitalize()
