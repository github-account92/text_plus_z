"""Get synthetic data into the w2l interface."""
import os
import sys

import numpy as np
from w2l.utils.data import read_data_config
from w2l.utils.vocab import make_vocab, parse_vocab
from .text import text_gen
from .audio import make_frequency_table, make_length_params, sonify


def fulfill_config_synth(config_path):
    """Check whether the data for a config exists and creates it if not.

    This is pretty dumb. It will check for existence of corpus csv, array
    directory and vocabulary file and create them if not found. This will not
    catch the case where these paths exist but contain garbage.

    Parameters:
        config_path: Path to config csv.

    """
    data_config = read_data_config(config_path)
    csv_path, array_dir, vocab_path = (data_config["csv_path"],
                                       data_config["array_dir"],
                                       data_config["vocab_path"])

    if not os.path.exists(csv_path):
        print("The requested corpus csv {} does not seem to exist. "
              "Creating...".format(csv_path))
        make_corpus_csv(csv_path)

    if not os.path.exists(vocab_path):
        print("The requested vocabulary file {} does not seem to exist. "
              "Creating...".format(vocab_path))
        make_vocab(csv_path, vocab_path)
    ch_ind, _ = parse_vocab(vocab_path)

    if not os.path.isdir(array_dir):
        create_data_dir = input("The requested data directory {} does not "
                                "seem to exist. Do you want to create it? "
                                "This could take a very long time. Type y/n "
                                "(no exits the program):".format(array_dir))
        if create_data_dir.lower() == "y":
            synthesize_audio(array_dir, ch_ind, csv_path)
        else:
            sys.exit("Data directory does not exist and creation not "
                     "requested.")


def make_corpus_csv(out_path):
    """Create a csv containing corpus info for synthetic data.

    The data itself is also created here. ;)

    The csv will contain lines as follows:
        id, path, transcription, set
            id: File id. Numbers separated by dashes. SpeakerBook-Sequence.
                Uniquely identifies the origin .flac file, but you will want to
                combine it with the set to find it quickly.
                We only use one speaker and book per subset.
            path: Relative path to the file from corpus directory. Superfluous
                  given set and id, but some functions need the full info so
                  here it is.
            transcription: The text.
            set: Which subset this came from. We only use s_train and s_test.

    Parameters:
        out_path: Path you want the corpus csv to go to.

    """
    print("Creating {} from synthetic data...".format(out_path))

    corpora = ["s_train", "s_test"]
    examples_per_corpus = [200000, 20000]
    gen = text_gen(maxlen=5)

    with open(out_path, mode="w") as corpus_csv:
        for corpus, samples in zip(corpora, examples_per_corpus):
            print("\tCreating {}...".format(corpus))

            for ind in range(1, samples + 1):
                transcr = next(gen)
                fid = "-".join([corpus, corpus, str(ind).zfill(6)])
                corpus_csv.write(
                    ",".join([fid, "dummy", transcr, corpus]) + "\n")

                if not ind % 1000:
                    print("\t\tCreated {} texts...".format(ind))


def synthesize_audio(array_dir, vocab, csv_path):
    """Synthesize signals for a given corpus csv.

    Parameters:
        array_dir: Path to directory where the created arrays should be stored.
        vocab: Dict mapping characters to indices.
        csv_path: Path to the corpus csv.

    """
    print("Synthesizing audio in {}...".format(array_dir))
    if not os.path.isdir(array_dir):
        os.makedirs(array_dir)

    freqs = make_frequency_table(vocab=vocab)
    len_means, len_stds = make_length_params(vocab)
    with open(csv_path) as corpus_csv:
        for n, line in enumerate(corpus_csv, start=1):
            fid, _, transcr, _ = line.strip().split(",")
            signal, segmentation = sonify(transcr, freqs, len_means, len_stds,
                                          vocab)
            np.save(os.path.join(array_dir, fid + ".npy"), signal)
            np.save(os.path.join(array_dir, fid + "_segment.npy"),
                    segmentation)

            if not n % 1000:
                print("\tCreated {} signals...".format(n))
