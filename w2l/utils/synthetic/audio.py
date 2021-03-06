"""Create synthetic signals for given synthetic text."""
import numpy as np
from scipy.signal import savgol_filter


def make_frequency_table(vocab, channels=20, mean=-3, std=5, seed=1):
    """Create random dominant frequencies for each character.

    We simply draw samples from a Gaussian distribution and truncate at 0.

    Parameters:
        vocab: List of characters.
        channels: Number of channels to sample from.
        mean: Mean for the random samples.
        std: Standard deviation for the random samples.
        seed: Seed for random numbers.

    Returns:
        2D numpy table containing frequencies for one character per row.

    """
    np.random.seed(seed)

    frequency_weights = std*np.random.randn(len(vocab), channels) + mean
    frequency_weights = np.clip(frequency_weights, 0, None).astype(np.float32)
    return frequency_weights


def make_length_params(vocab, minlen=5, maxlen=12, seed=1):
    """Create 'typical' pronunciation lengths for each character.

    For each character we draw a mean and an std that will be used each time
    when the character is sampled.

    Parameters:
        vocab: List of characters.
        minlen: Minimum mean length that may be drawn.
        maxlen: Maximum mean length that may be drawn.
        seed: Random seed.

    Return:
        Two 1D arrays containing means and stds respectively.

    """
    np.random.seed(seed)

    means = np.random.uniform(low=minlen, high=maxlen, size=len(vocab))
    stds = 2*np.random.rand(len(vocab))
    return means, stds


def sonify(utterance, freq_table, length_means, length_stds, vocab,
           noise_std=0.3, silence_mean=9, noise_prob=0.05, seed=1):
    """Turn a piece of written text into a signal.

    Also gives a per-frame segmentation of the underlying latents (characters,
    noise etc.)

    Parameters:
        utterance: The piece of text, string.
        freq_table: 2D numpy array containing in each row the frequency
                    representation of the corresponding character.
        length_means: 1D numpy array containing the mean length for each
                      character.
        length_stds: 1D numpy array containing the length std per character.
        vocab: Dict mapping characters to indices.
        noise_std: Standard deviation for noise that is put over the whole
                   signal.
        silence_mean: Mean length of silence periods.
        noise_prob: Probability that, after any character, either silence or
                    "disruptive" noise is injected.
        seed: Numpy random seed.

    Returns:
        Tuple of signal (2d, channels x time) and segmentation (1d, time).

    """
    np.random.seed(seed)

    channels = freq_table.shape[1]

    def make_silence():
        return np.zeros((channels, np.round(np.random.randn() +
                                            silence_mean).astype(np.int32)),
                        dtype=np.float32)

    def make_noise():
        return 10*noise_std*np.random.randn(
            channels, np.round(
                np.random.randn() + silence_mean//2).astype(np.int32)).astype(
            np.float32) + 1.5

    audio = [make_silence()]
    segmentation = [-1] * audio[0].shape[-1]  # -1 codes silence
    for char in utterance:
        if np.random.rand() < noise_prob:
            if np.random.rand() < 0.5:
                audio.append(make_noise())
                segmentation += [-2] * audio[-1].shape[1]  # -2 codes noise
            else:
                audio.append(make_silence())
                segmentation += [-1] * audio[-1].shape[1]

        pron_length = (length_stds[vocab[char]]*np.random.randn() +
                       length_means[vocab[char]])
        pron_length = np.round(pron_length).astype(np.int32)
        pron = np.tile(freq_table[vocab[char]][:, np.newaxis],
                       [1, pron_length])

        audio.append(pron)
        segmentation += [vocab[char]] * pron_length
    audio.append(make_silence())
    segmentation += [-1] * audio[-1].shape[1]

    audio = np.concatenate(audio, axis=1)
    smoothened = savgol_filter(audio, polyorder=1, window_length=5)
    smoothened += noise_std*np.random.randn(
        *smoothened.shape).astype(np.float32)
    smoothened = np.clip(smoothened, 0, None)
    return smoothened, np.asarray(segmentation, dtype=np.int32)
