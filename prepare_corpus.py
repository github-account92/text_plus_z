"""Prepare everything for a given data config."""
import argparse
import os
import sys

import librosa
import numpy as np

from w2l.utils.data import read_data_config, raw_to_mel
from w2l.utils.synthetic.files import fulfill_config_synth
from w2l.utils.vocab import make_vocab


def fulfill_config(corpus_path, config_path):
    """Check whether the data for a config exists and creates it if not.

    This is pretty dumb. It will check for existence of corpus csv, array
    directory and vocabulary file and create them if not found. This will not
    catch the case where these paths exist but contain garbage.

    NOTE: Creating a csv for the German corpus is currently not supported and
          not handled. It will likely crash due to directories not existing.
          Please use the csv in
          /data/corpora/German/corpus-without-positional.csv.

    Parameters:
        corpus_path: Path to corpus, e.g. /data/LibriSpeech or
        /data/corpora/German.
        config_path: Path to config csv.

    """
    data_config = read_data_config(config_path)
    csv_path, array_dir, vocab_path, data_type = (data_config["csv_path"],
                                                  data_config["array_dir"],
                                                  data_config["vocab_path"],
                                                  data_config["data_type"])
    n_freqs, window_size, hop_length, resample_rate = \
        (data_config["n_freqs"],
         data_config["window_size"],
         data_config["hop_length"],
         data_config["resample_rate"])
    keep_phase = data_config["keep_phase"]
    if resample_rate == 0:
        resample_rate = None

    if not os.path.exists(csv_path):
        print("The requested corpus csv {} does not seem to exist. "
              "Creating...".format(csv_path))
        make_corpus_csv(corpus_path, csv_path)

    if not os.path.exists(vocab_path):
        print("The requested vocabulary file {} does not seem to exist. "
              "Creating...".format(vocab_path))
        make_vocab(csv_path, vocab_path)

    if not os.path.isdir(array_dir):
        create_data_dir = input("The requested data directory {} does not "
                                "seem to exist. Do you want to create it? "
                                "This could take a very long time. Type y/n "
                                "(no exits the program):".format(array_dir))
        if create_data_dir.lower() == "y":
            preprocess_audio(csv_path, corpus_path, array_dir, data_type,
                             n_freqs, window_size, hop_length, resample_rate,
                             keep_phase)
        else:
            sys.exit("Data directory does not exist and creation not "
                     "requested.")


def make_corpus_csv(librispeech_path, out_path):
    """Create a csv containing corpus info.

    The csv will contain lines as follows:
        id, path, transcription, set
            id: File id. Numbers separated by dashes. Speaker-Book-Sequence.
                Uniquely identifies the origin .flac file, but you will want to
                combine it with the set to find it quickly.
            path: Relative path to the file from corpus directory. Superfluous
                  given set and id, but some functions need the full info so
                  here it is.
            transcription: The text.
            set: Which subset this came from, e.g. train-clean-360, dev-other.

    Parameters:
        librispeech_path: Path to LibriSpeech corpus, e.g. /data/LibriSpeech.
        out_path: Path you want the corpus csv to go to.

    """
    print("Creating {} from {}...".format(out_path, librispeech_path))

    corpora = ["train-clean-100", "train-clean-360", "train-other-500",
               "dev-clean", "dev-other", "test-clean", "test-other"]
    with open(out_path, mode="w") as corpus_csv:
        for corpus in corpora:
            print("\tProcessing {}...".format(corpus))
            corpus_walker = os.walk(os.path.join(librispeech_path, corpus))
            for path, _, files in corpus_walker:
                if not files:  # not a data directory
                    continue

                files.sort()  # puts transcriptions at the end
                transcrs = open(os.path.join(path, files[-1])).readlines()
                transcrs = [" ".join(t.strip().split()[1:]).lower()
                            for t in transcrs]
                if len(files[:-1]) != len(transcrs):
                    raise ValueError("Discrepancy in {}: {} audio files "
                                     "found, but {} transcriptions (should be "
                                     "the same).".format(
                                        path, len(files[:-1]), len(transcrs)))

                for f, t in zip(files[:-1], transcrs):
                    # file ID, relative path, transcription, corpus
                    fid = f.split(".")[0]
                    fpath = os.path.join(corpus, path, f)
                    corpus_csv.write(",".join([fid, fpath, t, corpus]) + "\n")


def preprocess_audio(csv_path, corpus_path, array_dir, data_type, n_freqs,
                     window_size, hop_length, resample_rate, keep_phase):
    """Preprocess many audio files with requested parameters.

    Parameters:
        csv_path: Path to corpus csv.
        corpus_path: Path to corpus, e.g. /data/LibriSpeech or
                     /data/corpora/German.
        array_dir: Path to directory where all the processed arrays should be
                   stored in.
        data_type: One of 'raw' or 'mel'. Whether to apply mel spectrogram
                   transformation. If 'raw', n_freqs, window_size and
                   hop_length are ignored.
        n_freqs: Number of mel frequencies to use.
        window_size: STFT window size.
        hop_length: STFT hop length.
        resample_rate: int. Hz to resample data to. If not given, no resampling
                       is performed and any sample rate != 16000 leads to a
                       crash.
        keep_phase: If set, keep the phase angle of the linear spectrogram and
                    append it to the channels.

    """
    os.mkdir(array_dir)
    with open(csv_path) as corpus_csv:
        for n, line in enumerate(corpus_csv, start=1):
            fid, fpath, _, _ = line.strip().split(",")
            path = os.path.join(corpus_path, fpath)
            audio, sr = librosa.load(path, sr=resample_rate)
            if not resample_rate and sr != 16000:
                raise ValueError("Sampling rate != 16000 ({}) found in {} "
                                 "with no resampling "
                                 "requested!".format(sr, path))
            if data_type == "mel":
                audio = raw_to_mel(audio, sr, window_size, hop_length, n_freqs,
                                   keep_phase)
            else:  # data_type == "raw"
                audio = audio[np.newaxis, :]  # add channel axis

            np.save(os.path.join(array_dir, fid + ".npy"),
                    audio.astype(np.float32))
            if not n % 1000:
                print("Processed {}...".format(n))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="See the file for info. :)")
    parser.add_argument("corpus_path",
                        help="Base path to corpus, e.g. /data/LibriSpeech or "
                             "/data/corpora/German.")
    parser.add_argument("config_path",
                        help="Path to config csv.")
    parser.add_argument("-S", "--synthetic",
                        action="store_true",
                        help="Set this flag to use synthetic data instead. In "
                             "this case, corpus_path will be ignored.")
    args = parser.parse_args()

    if args.synthetic:
        fulfill_config_synth(args.config_path)
    else:
        fulfill_config(args.corpus_path, args.config_path)
