import editdistance


def letter_error_rate_corpus(targets, predictions):
    """Compare two lists of strings to get the average letter error rate.

    LER is computed as the Levenshtein distance between the true and predicted
    string. Results are micro-averaged.
    """
    total_error_count = 0
    total_expected_count = 0
    for true, predicted in zip(targets, predictions):
        total_error_count += editdistance.eval(true, predicted)
        total_expected_count += len(true)
    return total_error_count / total_expected_count


def word_error_rate_corpus(targets, predictions):
    """Compare to lists of strings to get the average word error rate.

    WER is computed as the Levenshtein distance between the true and predicted
    string, with words being used as 'atomic" units, i.e. the string is split at
    whitespace Results are micro-averaged.
    """
    word_targets = [true.split() for true in targets]
    word_predictions = [predicted.split() for predicted in predictions]
    return letter_error_rate_corpus(word_targets, word_predictions)


def letter_error_rate_single(true, predicted):
    """Compare two strings and compute the letter error rate."""
    return letter_error_rate_corpus([true], [predicted])


def word_error_rate(true, predicted):
    """Compare two strings and compute the word error rate."""
    return word_error_rate_corpus([true], [predicted])
