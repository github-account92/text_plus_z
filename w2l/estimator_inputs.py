"""Create input functions for tf.Estimator models."""
import os

import numpy as np
import tensorflow as tf

from w2l.utils.data import raw_to_mel
from w2l.utils.rejects import GERMAN_REJECTS


###############################################################################
# Functions based off of log mel spectrograms stored as .npy arrays on disk.
###############################################################################
def w2l_input_fn_npy(csv_path, array_base_path, which_sets, train, vocab,
                     n_freqs, batch_size, threshold):
    """Build a tf.estimator input function for preprocessed data.

    NOTE: The data on disk is assumed to be stored channels_first.

    Parameters:
        csv_path: Should be the path to an appropriate csv file linking sound
                  files and label data. This csv should contain lines as
                  follows:
                    file_id,file_path,transcription,set where:
                      file_id: unique identifier for the file.
                      file_path: Relative path to the file within corpus
                                 directory. This is not needed here but is
                                 originally used to create the array directory,
                                 so this entry is assumed to be in the csv.
                      transcription: The target string.
                      set: Train/dev/test split.
        array_base_path: Base path to where the .npy arrays are stored.
        which_sets: Iterable (e.g. list, tuple or set) that contains all the
                    subsets to be considered (e.g. train-clean-360 etc.).
        train: Whether to shuffle and repeat data.
        vocab: Dictionary mapping characters to indices.
        n_freqs: Number of frequencies/"channels" in the data that will be
                 loaded. Needs to be set so that the model has this
                 information.
        batch_size: How big the batches should be.
        threshold: Float to use for thresholding input arrays.
                   See the _pyfunc further below for some important notes.

    Returns:
        get_next op of iterator.

    """
    print("Building input function for {} set using file {}...".format(
        which_sets, csv_path))
    # first read the csv and keep the useful stuff
    with open(csv_path, mode="r") as corpus:
        lines_split = [line.strip().split(",") for line in corpus]
    print("\t{} entries found.".format(len(lines_split)))

    print("\tFiltering requested subset...")
    lines_split = [line[:3] for line in lines_split if line[3] in which_sets]
    lines_split = [line for line in lines_split
                   if line[0] not in GERMAN_REJECTS]
    if not lines_split:
        raise ValueError("Filtering resulted in size-0 dataset! Maybe you "
                         "specified an invalid subset? You supplied "
                         "'{}'.".format(which_sets))
    print("\t{} entries remaining.".format(len(lines_split)))

    print("\tCreating the dataset...")
    ids, _, transcrs = zip(*lines_split)
    files = [os.path.join(array_base_path, fid + ".npy") for fid in ids]

    def _to_arrays(fname, trans):
        return _pyfunc_load_arrays_map_transcriptions(
            fname, trans, vocab, threshold)

    def gen():  # dummy to be able to use from_generator
        for file_name, transcr in zip(files, transcrs):
            # byte encoding is necessary in python 3, see TF 1.4 known issues
            yield file_name.encode("utf-8"), transcr.encode("utf-8")

    with tf.variable_scope("input"):
        data = tf.data.Dataset.from_generator(gen, (tf.string, tf.string))

        # TODO: try out bucketing
        if train:
            # this basically shuffles the full dataset
            data = data.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size=2 ** 18))

        output_types = [tf.float32, tf.int32, tf.int32, tf.int32]
        data = data.map(
            lambda fid, trans: tuple(tf.py_func(
                _to_arrays, [fid, trans], output_types,
                stateful=False)),
            num_parallel_calls=3)
        # NOTE 1: padding value of 0 for element 1 and 3 is just a dummy (since
        #         sequence lengths are always scalar)
        # NOTE 2: changing padding value of -1 for element 2 requires changes
        # in the model as well!
        pad_shapes = ((n_freqs, -1), (), (-1,), ())
        pad_values = (np.log(1e-10).astype(np.float32), 0, -1, 0)
        data = data.padded_batch(
            batch_size, padded_shapes=pad_shapes, padding_values=pad_values)
        data = data.map(pack_inputs_in_dict, num_parallel_calls=3)
        data = data.prefetch(2)  # 2 batches

        # build iterator
        print("\tBuilding iterator...")
        iterator = data.make_one_shot_iterator()
        return iterator.get_next()


def _pyfunc_load_arrays_map_transcriptions(file_name, trans, vocab, threshold):
    """Map file names to numpy arrays.

    Goes from file_id, transcriptions to a tuple np_array, coded_transcriptions
    (integers).
    Lengths are returned as well so they are known after padded batching.
    NOTE: Files are assumed to be stored channels_first. If this is not the
          case, this will cause trouble down the line!!

    Parameters:
        file_name: Path built from ID taken from data csv, should match npy
                   file names. Expected to be utf-8 encoded as bytes.
        trans: Transcription. Also utf-8 bytes.
        vocab: Dictionary mapping characters to integers.
        threshold: Float to use for thresholding the array. Any values more
                   than this much under the maximum will be clipped. E.g. if
                   the max is 15 and the threshold is 50, any value below -35
                   would be clipped to -35. It is your responsibility to pass a
                   reasonable value here -- this can vary heavily depending on
                   the scale of the data (it is however invariant to shifts).
                   Passing 0 or any "False" value here disables thresholding.
                   NOTE: You probably don't want to use this with
                   pre-normalized data since in that case, each example is
                   essentially on its own scale (one that results in mean 0 and
                   std 1, or whatever normalization was used) so a single
                   threshold value isn't really applicable.

    Returns:
        Tuple of 2D numpy array (n_freqs x seq_len), scalar (is seq_len),
        1D array (label_len), scalar (is label_len)

    """
    array = np.load(file_name.decode("utf-8"))
    # we make sure arrays are even in the time axis because otherwise there can
    # be trouble with reconstruction shapes due to strides
    if array.shape[-1] % 2:
        array = np.pad(array, pad_width=((0, 0), (0, 1)), mode="constant")
    length = np.int32(array.shape[-1])

    trans_mapped = np.array([vocab[ch] for ch in trans.decode("utf-8")],
                            dtype=np.int32)
    trans_length = np.int32(len(trans_mapped))

    if threshold:
        clip_val = np.max(array) - threshold
        array = np.maximum(array, clip_val)

    return array.astype(np.float32), length, trans_mapped, trans_length


###############################################################################
# To go straight from some collection of numpy arrays
###############################################################################
def w2l_input_fn_from_container(array_container, n_freqs, vocab_size,
                                bottleneck, is_mel=False):
    """Build input function from a container with 1D numpy arrays.

    Parameters:
        array_container: Should be something like a list of numpy arrays.
                         The elements of this container will continuously be
                         yielded. You need to modify this container from
                         outside to process more sequences. Only really
                         suitable for for estimator.predict.
                         The container needs to have at least three elements:
                         0: Input audio sequence (raw audio, is converted to mel
                            internally OR interpreted as already being mel if
                            is_mel is True).
                         1: Corresponding transcription (coded as indices!).
                            You may pass a dummy here if it's not available/not
                            interesting.
                         2: Latent sample. Only used if "only_decode" mode is
                            used. Again, pass dummy otherwise.

                         Any further elements are ignored.
        n_freqs: Frequencies to be expected in data.
        vocab_size: Duh.
        bottleneck: Size of the non-logit latent space.
        is_mel: If True, audio input (container[0]) is assumed to already be a
                pre-processed mel spectrogram. Otherwise the transformation is
                done in here.

    Returns:
        get_next op of iterator.

    """
    def gen():
        while True:
            if is_mel:
                as_mel = array_container[0]
            else:
                as_mel = raw_to_mel(
                    array_container[0], 16000, 400, 160, 128,
                    keep_phase=True).astype(np.float32)
            yield (as_mel, np.int32(as_mel.shape[-1]), array_container[2],
                   array_container[1], np.int32(array_container[1].shape[-1]))

    with tf.variable_scope("input"):
        data = tf.data.Dataset.from_generator(
            gen, (tf.float32, tf.int32, tf.float32, tf.int32, tf.int32))
        data = data.padded_batch(
            1,
            padded_shapes=((n_freqs, -1), (), (vocab_size+bottleneck, -1),
                           (-1,), ()),
            padding_values=(np.log(1e-10).astype(np.float32), 0, 0.,
                            -1, 0))
        data = data.map(pack_inputs_in_dict_cont, num_parallel_calls=3)

        # build iterator
        print("\tBuilding iterator...")
        iterator = data.make_one_shot_iterator()
        return iterator.get_next()


###############################################################################
# Helpers
###############################################################################
def pack_inputs_in_dict(audio, length, trans, trans_length):
    """For estimator interface (only allows one input -> pack into dict).

    Do note that the labels are also packed into the feature dict and None is
    passed for the label tensor!!
    """
    return ({"audio": audio, "audio_length": length,
            "transcription": trans, "trans_length": trans_length}, 0)


def pack_inputs_in_dict_cont(audio, length, latent, trans, trans_length):
    """For estimator interface (only allows one input -> pack into dict)."""
    return ({"audio": audio, "audio_length": length, "latent": latent,
             "transcription": trans, "trans_length": trans_length}, 0)
