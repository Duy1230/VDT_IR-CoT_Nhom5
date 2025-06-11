import re
import string
from collections import Counter


def normalize_answer_text(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Calculates the F1 score between a prediction and a ground truth answer.
    """
    norm_prediction = normalize_answer_text(prediction)
    norm_ground_truth = normalize_answer_text(ground_truth)

    if not norm_prediction or not norm_ground_truth:
        return 0.0

    prediction_tokens = norm_prediction.split()
    ground_truth_tokens = norm_ground_truth.split()

    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
