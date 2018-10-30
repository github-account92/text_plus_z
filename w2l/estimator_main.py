"""Train, run or evaluate w2l models."""
import os

import tensorflow as tf

from .utils.data import (read_data_config, parse_corpus_csv,
                         checkpoint_iterator)
from .utils.errors import letter_error_rate_corpus, word_error_rate_corpus
from .utils.vocab import parse_vocab
from .estimator_inputs import w2l_input_fn_npy, w2l_input_fn_from_container
from .estimator_model import w2l_model_fn


def run_asr(mode,
            data_config,
            model_config,
            model_dir,
            act="relu",
            ae_coeff=0.,
            batchnorm=True,
            blank_coeff=0.,
            bottleneck=15,
            data_format="channels_first",
            full_vae=False,
            mmd=False,
            phase=False,
            random=0.,
            reg=0.,
            topk=0,
            use_ctc=True,
            adam_params=(1e-4, 0.9, 0.9, 1e-8),
            batch_size=16,
            clipping=500,
            fix_lr=False,
            momentum=False,
            steps=500000,
            threshold=0.,
            verbose_losses=False,
            vis=100,
            which_sets=None,
            container=None,
            only_decode=False):
    """
    Execute generic ASR function.

    All of these parameters can be passed from w2l_cli. Please check that one
    for docs on what they are.
    Exception #1 is 'container' which is used only in "container mode".
    Exception #2 is only_decode, which should only ever be used in container
    mode because the other modes don't provide an appropriate input function.

    Returns:
        Depends on mode!
        train: Nothing is returned.
        eval-current: Returns the dict with evaluation results.
        eval-all: Returns a dict mapping checkpoint names to evaluation dicts.
        predict: Returns a generator over predictions for the requested set.
        return: Return the estimator object. Use this if you want access to
                   the variables or their values, for example.
        container: Returns a generator over predictions for the given
                      container.

    """
    # Set up, verify arguments etc.
    tf.logging.set_verbosity(tf.logging.INFO)

    data_config_dict = read_data_config(data_config)
    csv_path, array_dir, vocab_path, n_freqs = (
        data_config_dict["csv_path"], data_config_dict["array_dir"],
        data_config_dict["vocab_path"], data_config_dict["n_freqs"])
    # if we have phase in the data, the input function needs to know about the
    # additional channels
    if data_config_dict["keep_phase"]:
        phase_freqs_in_data = data_config_dict["window_size"] // 2 + 1
        n_freqs += phase_freqs_in_data
    else:
        phase_freqs_in_data = 0

    if act == "elu":
        act_fn = tf.nn.elu
    elif act == "relu":
        act_fn = tf.nn.relu
    else:  # swish -- since no other choice is allowed
        def act_fn(x): return x * tf.nn.sigmoid(x)

    # set up model
    ch_to_ind, ind_to_ch = parse_vocab(vocab_path)
    ind_to_ch[-1] = "<PAD>"

    params = {"model_config": model_config,
              "vocab_size": len(ch_to_ind),
              "act": act_fn,
              "use_bn": batchnorm,
              "data_format": data_format,
              "adam_args": adam_params,
              "clipping": clipping,
              "vis": vis,
              "reg": reg,
              "momentum": momentum,
              "fix_lr": fix_lr,
              "mmd": mmd,
              "bottleneck": bottleneck,
              "use_ctc": use_ctc,
              "ae_coeff": ae_coeff,
              "only_decode": only_decode,
              "phase": phase,
              "phase_freqs_in_data": phase_freqs_in_data,
              "topk": topk,
              "random": random,
              "full_vae": full_vae,
              "verbose_losses": verbose_losses,
              "blank_coeff": blank_coeff}
    # we set infrequent "permanent" checkpoints
    # we also disable the default SummarySaverHook IF profiling is requested
    config = tf.estimator.RunConfig(keep_checkpoint_every_n_hours=6,
                                    save_summary_steps=None if vis else 100,
                                    model_dir=model_dir)
    multi_gpu_model_fn = tf.contrib.estimator.replicate_model_fn(w2l_model_fn)
    estimator = tf.estimator.Estimator(model_fn=multi_gpu_model_fn,
                                       params=params, config=config)
    if mode == "return":
        return estimator

    # if not return, set up corresponding inputs and do the requested thing
    # TODO: some kind of switch between LibriSpeech and German subsets
    if not which_sets:
        if mode == "train":
            which_sets = ["train-clean-100", "train-clean-360",
                          "train-other-500", "dev-clean", "dev-other",
                          "training"]
        else:  # predict or eval -- use the test set
            which_sets = ["test-clean", "test-other", "test"]

    if mode == "container":
        if not container:
            raise ValueError("Container mode requested, but there is nothing "
                             "in the container. Please pass at least a dummy. "
                             "E.g. container = "
                             "[np.zeros(1000, dtype=np.float32)].")

        def input_fn():
            return w2l_input_fn_from_container(
                container, n_freqs, len(ch_to_ind) + 1, bottleneck)
    else:
        def input_fn():
            return w2l_input_fn_npy(
                csv_path, array_dir, which_sets, train=mode == "train",
                vocab=ch_to_ind, n_freqs=n_freqs, batch_size=batch_size,
                threshold=threshold)

    if mode == "train":
        estimator.train(input_fn=input_fn, steps=steps)
        return

    elif mode == "predict" or mode == "errors" or mode == "container":
        def gen():
            corpus = parse_corpus_csv(csv_path, which_sets)
            for ind, (prediction, (fid, transcr, speaker)) in enumerate(
                    zip(estimator.predict(input_fn=input_fn), corpus)):

                predictions_repacked = dict()
                if mode != "container":
                    predictions_repacked["fid"] = fid
                    predictions_repacked["true"] = transcr
                    predictions_repacked["speaker"] = speaker

                if use_ctc:
                    decoded = prediction["decoding"]
                    # remove padding and convert to chars
                    decoded = [[ind for ind in candidate if ind != -1]
                               for candidate in decoded]
                    decoded_char = ["".join([ind_to_ch[ind]
                                             for ind in candidate])
                                    for candidate in decoded]
                    predictions_repacked["decoding"] = decoded_char

                # construct a sorted list of layers and their activations
                # we separate encoder and decoder and also inputs, logits and
                # latent activations
                encoder_layers = [(n, a) for (n, a) in prediction.items() if
                                  leading_string(n) == "encoder_layer"]
                encoder_layers.sort(key=lambda tup: trailing_num(tup[0]))
                decoder_layers = [(n, a) for (n, a) in prediction.items() if
                                  leading_string(n) == "decoder_layer"]
                decoder_layers.sort(key=lambda tup: trailing_num(tup[0]))

                predictions_repacked["encoder_layers"] = encoder_layers
                predictions_repacked["decoder_layers"] = decoder_layers

                other_keys = ["logits", "latent", "input", "reconstruction",
                              "input_length"]
                if random:
                    other_keys += ["latent_means", "latent_logvar"]
                    if full_vae:
                        other_keys += ["logits_means", "logits_logvar"]
                for key in other_keys:
                    predictions_repacked[key] = prediction[key]
                yield predictions_repacked

        if mode == "predict" or mode == "container":
            return gen()

        if mode == "errors":
            # preferable to TF edit distance due to properly weighted averaging
            true = []
            predicted = []
            for p in gen():
                true.append(p["true"])
                predicted.append(p["decoding"][0])
            ler = letter_error_rate_corpus(true, predicted)
            wer = word_error_rate_corpus(true, predicted)
            print("LER: {}\nWER: {}".format(ler, wer))

            return ler, wer

    elif mode == "eval-all":
        eval_dicts = dict()
        for ckpt in checkpoint_iterator(model_dir):
            print("Evaluating checkpoint {}...".format(ckpt))
            eval_results = estimator.evaluate(
                input_fn=input_fn,
                checkpoint_path=os.path.join(model_dir, ckpt))
            print("Evaluation results:\n", eval_results)
            eval_dicts[ckpt] = eval_results
        return eval_dicts

    elif mode == "eval-current":
        eval_results = estimator.evaluate(input_fn=input_fn)
        print("Evaluation results:\n", eval_results)
        return eval_results


###############################################################################
# Helpers
###############################################################################
def leading_string(string):
    """Split e.g. "layer104" into "layer", "104" and returns "layer"."""
    alpha = string.rstrip('0123456789')
    return alpha


def trailing_num(string):
    """Split e.g. "layer104" into "layer", "104" and returns int("104")."""
    alpha = string.rstrip('0123456789')
    num = string[len(alpha):]
    return int(num)
