import numpy as np


def make_language(vocab, upper_weight=10, seed=0):
    """Create a very simple (unigram) language model for a given vocabulary.

    Parameters:
        vocab: List of characters (elementary units of the language).
        upper_weight: The maximum weight a character can receive to follow
                      another. E.g. using 10, the most probable follow-up may
                      be up to 10 times as likely as the least likely (non-zero
                      one).
        seed: Seed for the random weights.

    Returns: Dict of dicts representing the language model.
    """
    np.random.seed(seed)
    language_model = np.random.randint(
        low=0, high=upper_weight,
        size=(len(vocab), len(vocab))).astype(np.float32)
    language_model[:, CH_IND[" "]] =+ 6
    language_model[CH_IND[" "], CH_IND[" "]] = 0.0  # space can't follow space
    language_model /= np.sum(language_model, axis=1, keepdims=True)

    return language_model


def make_sequence(lang_model, stop_prob=0.2, maxlen=0):
    """Create a single random sequence of our language.

    Parameters:
        lang_model: The language model.
        stop_prob: Probability to stop generating when a space character has
                   been selected.
        maxlen: Int; if given, stop generation once this many words (!) have
                been generated
    """
    # to start, pretend we're following space
    start_ind = np.random.choice(lang_model.shape[1],
                                 p=lang_model[CH_IND[" "]])
    seq = IND_CH[start_ind]
    while True:
        next_ind = np.random.choice(lang_model.shape[1],
                                    p=lang_model[CH_IND[seq[-1]]])
        if next_ind == CH_IND[" "]:
            stop = np.random.rand()
            if stop < stop_prob or (maxlen and len(seq.split()) >= maxlen):
                break
        seq += IND_CH[next_ind]
    return seq


def text_gen(maxlen=0):
    lang_mod = make_language(VOCAB)
    while True:
        yield make_sequence(lang_mod, maxlen=maxlen)


VOCAB = sorted(["a", "b", "c", "d", "e", " "])
CH_IND = dict(zip(VOCAB, range(len(VOCAB))))
IND_CH = {v: k for k, v in CH_IND.items()}
