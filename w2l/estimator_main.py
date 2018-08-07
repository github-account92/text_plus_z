import tensorflow as tf

from .utils.data import (read_data_config, extract_transcriptions_and_speaker,
                         checkpoint_iterator)
from .utils.errors import letter_error_rate_corpus, word_error_rate_corpus
from .utils.vocab import parse_vocab
from .estimator_inputs import (w2l_input_fn_npy, w2l_input_fn_from_container,
                               redo_repetitions)
from .estimator_model import w2l_model_fn


def run_asr(mode, data_config, model_config, model_dir,
            act="relu", ae_coeff=0., batchnorm=True, bottleneck=128,
            data_format="channels_first", mmd=False, reg=0.,
            use_ctc=True,
            adam_params=(1e-4, 0.9, 0.9, 1e-8), batch_size=16, clipping=500,
            fix_lr=False, momentum=False, normalize=False, steps=500000,
            threshold=0., vis=100, which_sets=None,
            container=None, only_decode=False):
    """
    All of these parameters can be passed from w2l_cli. Please check
    that one for docs on what they are.
    The exception is 'container' which is used only in "container mode".
    
    Returns:
        Depends on mode!
        If train, eval-current or eval-all: Nothing is returned.
        If predict: Returns a generator over predictions for the requested set.
        If return: Return the estimator object. Use this if you want access to
                   the variables or their values, for example.
        If container: Returns a generator over predictions for the given
                      container.
    """
    # TODO refactor
    # Set up, verify arguments etc.
    tf.logging.set_verbosity(tf.logging.INFO)

    # These used to be separate CLAs, but essentially "belong together" so this
    # results in less cluttered calls to the training script.
    data_config_dict = read_data_config(data_config)
    csv_path, array_dir, vocab_path, mel_freqs = (
        data_config_dict["csv_path"], data_config_dict["array_dir"],
        data_config_dict["vocab_path"], data_config_dict["n_freqs"])
    # if we have phase in the data, the input function needs to know about the
    # additional channels
    if data_config_dict["keep_phase"]:
        mel_freqs += data_config_dict["window_size"] // 2 + 1

    if act == "elu":
        act = tf.nn.elu
    elif act == "relu":
        act = tf.nn.relu
    else:  # swish -- since no other choice is allowed
        def act(x): return x * tf.nn.sigmoid(x)

    # set up model
    ch_to_ind, ind_to_ch = parse_vocab(vocab_path)
    ind_to_ch[-1] = "<PAD>"

    params = {"model_config": model_config,
              "vocab_size": len(ch_to_ind),
              "act": act,
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
              "only_decode": only_decode}
    # we set infrequent "permanent" checkpoints
    # we also disable the default SummarySaverHook IF profiling is requested
    config = tf.estimator.RunConfig(keep_checkpoint_every_n_hours=6,
                                    save_summary_steps=None if vis else 100,
                                    model_dir=model_dir)
    mutli_gpu_model_fn = tf.contrib.estimator.replicate_model_fn(w2l_model_fn)
    estimator = tf.estimator.Estimator(model_fn=mutli_gpu_model_fn,
                                       params=params,
                                       config=config)
    if mode == "return":
        return estimator

    # if not return, set up corresponding inputs and do the requested thing
    # TODO some kind of switch between LibriSpeech and German subsets
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
                             "np.zeros(1000, dtype=np.float32).")

        def input_fn():
            return w2l_input_fn_from_container(container, only_decode)
    else:
        def input_fn():
            return w2l_input_fn_npy(
                csv_path, array_dir, which_sets, train=mode == "train",
                vocab=ch_to_ind, n_freqs=mel_freqs, batch_size=batch_size,
                normalize=normalize, threshold=threshold)

    if mode == "train":
        estimator.train(input_fn=input_fn, steps=steps)
        return

    elif mode == "predict" or mode == "errors" or mode == "container":
        def gen():
            transcriptions, speakers = extract_transcriptions_and_speaker(
                csv_path, which_sets)
            for ind, (prediction, true, speaker) in enumerate(
                    zip(estimator.predict(input_fn=input_fn),
                        transcriptions, speakers)):

                predictions_repacked = dict()
                if mode != "container":
                    predictions_repacked["true"] = true
                    predictions_repacked["speaker"] = speaker
                predictions_repacked["input_length"] = prediction["input_length"]

                if use_ctc:
                    pred = prediction["decoding"]
                    # remove padding and convert to chars
                    pred = [[p for p in candidate if p != -1] for candidate in pred]
                    pred_ch = ["".join([ind_to_ch[ind] for ind in candidate])
                               for candidate in pred]
                    # pred_ch = [redo_repetitions(candidate) for candidate in pred_ch]
                    predictions_repacked["decoding"] = pred_ch

                # construct a sorted list of layers and their activations, with
                # input in front and output (logits) in the back
                layers = [(n, a) for (n, a) in prediction.items() if
                          leading_string(n) in ["encoder_layer", "decoder_layer", "block", "dense"]]
                layers.sort(key=lambda tup: trailing_num(tup[0]))
                layers.append(("logits", prediction["logits"]))
                layers.append(("latent", prediction["latent"]))
                layers.insert(0, ("input", prediction["input"]))

                predictions_repacked["all_layers"] = layers
                predictions_repacked["reconstruction"] = prediction["reconstruction"]
                yield predictions_repacked

    if mode == "predict" or mode == "container":
        return gen()

    if mode == "errors":
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
        for ckpt in checkpoint_iterator(model_dir):
            print("Evaluating checkpoint {}...".format(ckpt))
            eval_results = estimator.evaluate(input_fn=input_fn)
            print("Evaluation results:\n", eval_results)
        return

    elif mode == "eval-current":
        eval_results = estimator.evaluate(input_fn=input_fn)
        print("Evaluation results:\n", eval_results)
        return


###############################################################################
# Helpers
###############################################################################
def leading_string(string):
    """Splits e.g. "layer104" into "layer", "104" an returns "layer"."""
    alpha = string.rstrip('0123456789')
    return alpha


def trailing_num(string):
    """Splits e.g. "layer104" into "layer", "104" an returns int("104")."""
    alpha = string.rstrip('0123456789')
    num = string[len(alpha):]
    return int(num)
