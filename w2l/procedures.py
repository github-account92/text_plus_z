import os

import numpy as np

from .estimator_main import run_asr


def compute_all_latents(path, data_format, *args, **kwargs):
    """Compute or get all latent representations for the test set.

    Parameters:
        path: Path for the latents. If not already existing, computes all
              latents and stores them here. If existing, load them from here.
        data_format: channels_first or last.
        *args: Arguments passed to run_asr.
        **kwargs: Keyword arguments passed to run_asr.

    Returns:
        Dictionary mapping speaker ID to a list containing two elements.
        Element 1 is an n x v array containing all logits for the speaker.
        Element 2 is an n x d array containing all latents for the speaker.
    """
    if not os.path.isdir(path):
        os.mkdir(path)
        predictions = run_asr("predict", use_ctc=False, *args, **kwargs)
        store = {}
        print("Collecting all latent representations...")
        for ind, pr in enumerate(predictions, start=1):
            logits = pr["all_layers"][-2][1]
            latent = pr["all_layers"][-1][1]
            speaker = pr["speaker"]
            if speaker not in store:
                store[speaker] = [[logits], [latent]]
            else:
                store[speaker][0].append(logits)
                store[speaker][1].append(latent)
            if not ind % 500:
                print("Done with {}...".format(ind))
        for sp in store:
            store[sp][0] = np.concatenate(store[sp][0], axis=1 if data_format == "channels_first" else 0)
            store[sp][1] = np.concatenate(store[sp][1], axis=1 if data_format == "channels_first" else 0)
            if data_format == "channels_first":
                store[sp][0] = store[sp][0].transpose()
                store[sp][1] = store[sp][1].transpose()
            np.save(os.path.join(path, sp + "_logits.npy"), store[sp][0])
            np.save(os.path.join(path, sp + "_latent.npy"), store[sp][1])
    else:
        store = {}
        for file in os.listdir(path):
            sp, space = file.split("_")
            if sp not in store:
                store[sp] = [None, None]
            if space[:-4] == "logits":
                store[sp][0] = np.load(os.path.join(path, file))
            elif space[:-4] == "latent":
                store[sp][1] = np.load(os.path.join(path, file))
            else:
                raise ValueError("Invalid file name {}".format(file))

    return store


def speaker_averages(store):
    """Compute average latent vectors for each speaker.

    Parameters:
        store: Dict from compute_all_latents.

    Returns:
        Similar dict containing averages instead.
    """
    average_store = {}
    for sp in store:
        average_store[sp] = [None, None]
        average_store[sp][0] = np.mean(store[sp][0], axis=0)
        average_store[sp][1] = np.mean(store[sp][1], axis=0)
    return average_store
