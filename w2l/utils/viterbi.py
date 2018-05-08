"""Viterbi decoding. No Language Model!

    >>> import numpy as np
    >>> transitions = np.array([
    ...    [0, .3, .7],
    ...    [.2, .3, .5],
    ...    [0, 0, 1]
    ... ])
    >>> emissions = np.array([
    ...    [
    ...        [0, .2, .8],
    ...        [.4, .2, .1],
    ...        [0, .5, .5],
    ...        [.9, .1, 0],
    ...        [.6, .3, .1]
    ...    ]
    ... ])
    >>> decode(emissions, transitions)
    array([[2, 2, 2, 0, 0]])

To get rid of duplicates, do this:
    >>> [list(drop_reps(d)) for d in decode(emissions, transitions)]
    [[2, 0]]
"""
import numpy as np


def decode(emissions: np.array, transitions: np.array):
    """Return best path through viterbi trellis constructed from emissions and transitions.

    Assumes the following:
    - initial probabilities are equal for all characters, so not used here
    - emissions is a Batch x Timesteps x Vocabulary matrix
    - transitions is Batch x Vocabulary x Vocabulary matrix

    Returns a Batch x Timesteps array of character IDs.
    """
    _, n_timesteps, n_states = emissions.shape
    trellis = np.zeros_like(emissions)
    trellis[:, 0, :] = emissions[:, 0, :]
    for timestep in range(1, n_timesteps):
        for state in range(n_states):
            trans_prob = trellis[:, timestep - 1] + transitions[state]
            trellis[:, timestep, state] = trans_prob.max(axis=1)
            trellis[:, timestep, state] = (
                trellis[:, timestep, state] + emissions[:, timestep, state])
    return trellis.argmax(axis=2)


def drop_reps(seq: np.array):
    """Skips all repetitions of items in a sequence.

    Note that this is not batched and also a generator.
    """
    if not seq.any():
        return ()
    seq = iter(seq)
    # get first value
    prev_val = next(seq)
    yield prev_val
    for val in seq:
        if val != prev_val:
            prev_val = val
            yield val
