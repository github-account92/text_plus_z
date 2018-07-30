import os

import librosa
import numpy as np

from w2l.utils.rejects import GERMAN_REJECTS


DATA_CONFIG_EXPECTED_ENTRIES = {
    "csv_path", "array_dir", "vocab_path", "data_type", "n_freqs",
    "window_size", "hop_length", "normalize", "keep_phase"}
DATA_CONFIG_INT_ENTRIES = {"n_freqs", "window_size", "hop_length"}
DATA_CONFIG_BOOL_ENTRIES = {"normalize", "keep_phase"}


def read_data_config(config_path):
    """Read a config file with information about the data.
    
    The file should be in csv format and contain the following entries:
        csv_path: Path to a file like corpus.csv on Poseidon.
        array_dir: Path to the directory containing the corresponding numpy
                   arrays.
        vocab_path: Path to a vocabulary file such as one created by vocab.py.
        data_type: One of 'raw' or 'mel'.
        n_freqs: Frequencies (e.g. STFT or mel bins) to be expected in the
                 data. Will lead to problems if this does not match with
                 reality. If data_type is 'raw' this should be 1!
        window_size: Window size to use for STFT (n_fft argument in librosa).
                     Relevant for preprocessing only (and for you to know the
                     parameters of the data). Ignored if data_type is 'raw'.
        hop_length: STFT hop length. See window_size. Ignored if data_type is
                    'raw'.
        normalize: Whether to normalize data in preprocessing. True or False.
        keep_phase: If set, keep the phase angle of the linear spectrogram and
                    append it to the channels.
        
    Entries can be in any order. Missing or superfluous entries will result in
    a crash. You can add comments via lines starting with '#'.
    
    Returns:
        dict with config file entries. Numerical entries are converted to int.
    """
    config_dict = dict()
    with open(config_path) as data_config:
        for line in data_config:
            if line[0] == "#":
                continue
            key, val = line.strip().split(",")
            config_dict[key] = val
    found_entries = set(config_dict.keys())
    for f_entry in found_entries:
        if f_entry not in DATA_CONFIG_EXPECTED_ENTRIES:
            raise ValueError("Entry {} found in config file which should not "
                             "be there.".format(f_entry))
    for e_entry in DATA_CONFIG_EXPECTED_ENTRIES:
        if e_entry not in found_entries:
            raise ValueError("Entry {} expected in config file, but not "
                             "found.".format(e_entry))
    for i_entry in DATA_CONFIG_INT_ENTRIES:
        config_dict[i_entry] = int(config_dict[i_entry])

    def str_to_bool(string):
        if string == "True":
            return True
        elif string == "False":
            return False
        else:
            raise ValueError("Invalid bool string {}. Use 'True' or "
                             "'False'.".format(string))
    for b_entry in DATA_CONFIG_BOOL_ENTRIES:
        config_dict[b_entry] = str_to_bool(config_dict[b_entry])

    return config_dict


def extract_transcriptions_and_speaker(csv_path, which_sets):
    """Return a list of transcriptions and speakers from a corpus csv as strings.

    Parameters:
        csv_path: Path to corpus csv that has all the transcriptions.
        which_sets: Iterable (e.g. list, tuple or set) that contains all the
                    subsets to be considered (e.g. train-clean-360 etc.).

    Returns:
        Two lists of strings, the transcriptions and speakers (in order!).
    """
    with open(csv_path, mode="r") as corpus:
        lines = [line.strip().split(",") for line in corpus]
    lines = [line for line in lines if line[0] not in GERMAN_REJECTS]
    transcrs = [line[2] for line in lines if line[3] in which_sets]
    speakers = [line[0] for line in lines if line[3] in which_sets]
    speakers = [l.split("-")[0] for l in speakers]

    if not transcrs:
        raise ValueError("Filtering resulted in size-0 dataset! Maybe you "
                         "specified an invalid subset? You supplied "
                         "'{}'.".format(which_sets))
    return transcrs, speakers


def checkpoint_iterator(ckpt_folder):
    # TODO new estimator arguments can probably make this less hacky!
    """Iterates over checkpoints in order and returns them.

    This modifies the "checkpoint meta file" directly which might not be the
    smartest way to do it.
    Note that this file yields checkpoint names for convenience, but the main
    function is actually the modification of the meta file.

    Parameters:
        ckpt_folder: Path to folder that has all the checkpoints. Usually the
                     estimator's model_dir. Also needs to contain a file called
                     "checkpoint" that acts as the "meta file".

    Yields:
        Paths to checkpoints, in order.
    """
    # we store the original text to re-write it
    try:
        with open(os.path.join(ckpt_folder, "checkpoint")) as ckpt_file:
            next(ckpt_file)
            orig = ckpt_file.read()
    except:  # the file might be empty because reasons...
        orig = ""

    # get all the checkpoints
    # we can't rely on the meta file (doesn't store permanent checkpoints :()
    # so we check the folder instead.
    ckpts = set()
    for file in os.listdir(ckpt_folder):
        if file.split("-")[0] == "model.ckpt":
            ckpts.add(int(file.split("-")[1].split(".")[0]))
    ckpts = sorted(list(ckpts))
    ckpts = ["\"model.ckpt-" + str(ckpt) + "\"" for ckpt in ckpts]

    # fill them in one-by-one and leave
    for ckpt in ckpts:
        with open(os.path.join(ckpt_folder, "checkpoint"),
                  mode="w") as ckpt_file:
            ckpt_file.write("model_checkpoint_path: " + ckpt + "\n")
            ckpt_file.write(orig)
        yield ckpt


def raw_to_mel(audio, sampling_rate, window_size, hop_length, n_freqs,
               normalize, keep_phase=False):
    """Go from 1D numpy array containing audio waves to mel spectrogram.

    Parameters:
        audio: 1D numpy array containing the audio.
        sampling_rate: Sampling rate of audio.
        window_size: STFT window size.
        hop_length: Distance between successive STFT windows.
        n_freqs: Number of mel frequency bins.
        normalize: If set, normalize log power spectrogram to mean 0, std 1.
        keep_phase: If set, keep the phase angle of the linear spectrogram and
                    append it to the channels.

    Returns:
        Processed spectrogram.
    """
    spectro = librosa.stft(audio, n_fft=window_size, hop_length=hop_length,
                           center=True)
    power = np.abs(spectro)**2
    mel = librosa.feature.melspectrogram(S=power, sr=sampling_rate,
                                         n_mels=n_freqs)
    logmel = np.log(mel + 1e-11)
    if normalize:
        logmel = (logmel - np.mean(logmel)) / np.std(logmel)
    if keep_phase:
        phase_angle = np.angle(spectro)
        logmel = np.concatenate((logmel, phase_angle))
    return logmel
