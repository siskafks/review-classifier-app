import numpy as np

class MinMaxNormalization:
  def normalize(self, review_helpful, sentiment, subjectivity, word_count, noun_count, adj_count, verb_count, adv_count, authenticity, at):
    data = {
        'Review_helpful': review_helpful,
        'Sentiment': sentiment,
        'Subjectivity': subjectivity,
        'Word_Count': word_count,
        'Noun_Count': noun_count,
        'Adj_Count': adj_count,
        'Verb_Count': verb_count,
        'Adv_Count': adv_count,
        'Authenticity': authenticity,
        'AT': at
    }

    min_values = {
        'Review_helpful': 0,
        'Sentiment': -1,
        'Subjectivity': 0,
        'Word_Count': 1,
        'Noun_Count': 0,
        'Adj_Count': 0,
        'Verb_Count': 0,
        'Adv_Count': 0,
        'Authenticity': 0,
        'AT': -25
    }

    max_values = {
        'Review_helpful': 8975,
        'Sentiment': 1,
        'Subjectivity': 1,
        'Word_Count': 1541,
        'Noun_Count': 185,
        'Adj_Count': 91,
        'Verb_Count': 102,
        'Adv_Count': 48,
        'Authenticity': 1.667,
        'AT': 36
    }

    normalized_data = np.array([(data[key] - min_values[key]) / (max_values[key] - min_values[key]) for key in data]).reshape(1, -1)

    return normalized_data