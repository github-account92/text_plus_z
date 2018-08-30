import numpy as np

from .estimator_main import run_asr


def speaker_averages(data_format="channels_first", *args, **kwargs):
    """Compute average latent vectors for each speaker.

    Parameters:
        data_format: channels_first or last.
        args: Arguments passed to run_asr.
        kwargs: Keyword arguments passed to run_asr.
    """
    time_ax = 1 if data_format == "channels_first" else 0
    predictions = run_asr("predict", use_ctc=False, data_format=data_format,
                          *args, **kwargs)
    store = {}
    print("Collecting all latent representations...")
    for ind, pr in enumerate(predictions, start=1):
        logits = pr["all_layers"][-2][1]
        latent = pr["all_layers"][-1][1]
        speaker = pr["speaker"]
        if speaker not in store:
            store[speaker] = [1, np.mean(logits, axis=time_ax),
                              np.mean(latent, axis=time_ax)]
        else:
            store[speaker][0] += 1
            store[speaker][1] += np.mean(logits, axis=time_ax)
            store[speaker][2] += np.mean(latent, axis=time_ax)
        if not ind % 500:
            print("Done with {}...".format(ind))
    for sp in store:
        store[sp][1] /= store[sp][0]
        store[sp][2] /= store[sp][0]

    return store
