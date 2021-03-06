"""Create model functions for tf.Estimator."""
import numpy as np
import tensorflow as tf

from .utils.hooks import SummarySaverHookWithProfile
from .utils.layers import (conv_layer, transposed_conv_layer,
                           cnn1d_from_config, rnn_from_config)
from .utils.model import (decode, decode_top, dense_to_sparse, lr_annealer,
                          clip_and_step,
                          # feature_map_global_variance_regularizer,
                          feature_map_local_variance_regularizer,
                          blank_prob_loss, reconstruction_loss, mmd_loss)


def w2l_model_fn(features, labels, mode, params, config):
    """Model function for tf.estimator.

    Parameters:
        features: Should be a dict containing string keys:
            audio: batch_size x channels x seq_len tensor of input sequences.
                   Note: Must be channels_first!!
            audio_length: batch_size tensor of sequence lengths for each input.
          Due to REASONS, it should also include the labels via following
          keys:
            transcription: batch_size x label_len tensor of label indices
                           (standing for letters).
            trans_length: batch_size tensor of sequence lengths for each
                          transcription.
          If you don't have labels for your data, pass dummies anyway!
        labels: Ignored. Pass labels in features!!
        mode: Train, Evaluate or Predict modes from tf.estimator.
        params: Should be a dict with the following string keys:
            model_config: Base path to config files to build the encoder/
                          decoder *excluding* final layers.
            vocab_size: Size of the vocabulary, to get the size for the final
                        layer.
            act: The hidden activation function, e.g. tf.nn.relu or tf.nn.elu.
            use_bn: Bool, whether to use batch normalization.
            data_format: String, channels_first or otherwise assumed to be
                         channels_last (this is not checked here!!).
            adam_args: List with Adam parameters (in order!!).
            clipping: Float, to set the gradient clipping norm. 0 disables
                      clipping.
            vis: Int, whether to include visualizations besides loss and steps
                 per time and if so how often.
            reg: float coefficient for regularizer for latent space. 0 disables
                 it.
            momentum: Bool, if set use gradient descent with Nesterov momentum
                      instead of Adam.
            fix_lr: Bool, if set use the provided learning rate instead of
                    the automatically annealed one.
            mmd: Coefficient for MMD loss for latent space (Wasserstein AE).
                 0 to disable.
            bottleneck: Size of bottleneck (style space).
            use_ctc: Bool, whether to use CTC loss. If False, speech
                     recognition is not trained.
            ae_coeff: Coefficient for AE loss.
            only_decode: Bool, if set only run the decoder and assume that the
                         inputs are logits + style space samples.
            phase: If false, discard the phase in the input IF it is there in
                   the first place.
            phase_freqs_in_data: Int, how many phase frequencies are in the
                                 data. Should generally be windowsize/2 + 1 or
                                 just 0.
            topk: Int, if > 0 only keep information on the top k logits at each
                  time step.
            random: Float, if > 0 use a random encoder. This float represents
                    the coefficient for the L1 variance regularizer.
            full_vae: Bool; If set apply WAE stuff to logits as well (i.e. MMD
                      loss).
            verbose_losses: Bool; if set add summaries for losses that aren't
                            trained on.
            blank_coeff: Coefficient for blank label activity regularizer;
                         0 disables it.
            adversarial: Bool, use adversarial traning or return adversarial
                         gradients in predictions.
            l2_reg: Float, coefficient for kernel L2 regularizer. 0 disables it.
        config: RunConfig object passed through from Estimator.

    Returns:
        An EstimatorSpec to be used in tf.estimator.

    """
    # first get all the params for convenience
    model_config = params["model_config"]
    vocab_size = params["vocab_size"]
    act = params["act"]
    use_bn = params["use_bn"]
    data_format = params["data_format"]
    adam_args = params["adam_args"]
    clipping = params["clipping"]
    vis = params["vis"]
    reg_coeff = params["reg"]
    momentum = params["momentum"]
    fix_lr = params["fix_lr"]
    mmd_coeff = params["mmd"]
    bottleneck = params["bottleneck"]
    use_ctc = params["use_ctc"]
    ae_coeff = params["ae_coeff"]
    only_decode = params["only_decode"]
    phase = params["phase"]
    phase_freqs_in_data = params["phase_freqs_in_data"]
    topk = params["topk"]
    random = params["random"]
    full_vae = params["full_vae"]
    verbose_losses = params["verbose_losses"]
    blank_coeff = params["blank_coeff"]
    adversarial = params["adversarial"]
    l2_reg = params["l2_reg"]

    cf = data_format == "channels_first"

    # construct model input -> output
    audio, seq_lengths = features["audio"], features["audio_length"]
    if cf:
        n_channels = audio.shape.as_list()[1]
    else:
        n_channels = audio.shape.as_list()[-1]
    if not phase and phase_freqs_in_data:
        n_channels -= phase_freqs_in_data
        audio_with_phase = audio
        audio = audio[:, :n_channels, :]
    labels = features["transcription"]
    if not cf:
        audio = tf.transpose(audio, [0, 2, 1])

    with tf.variable_scope("model"):
        pre_out, total_stride, encoder_layers = read_apply_model_config(
            model_config + "_enc", audio, act=act, batchnorm=use_bn,
            train=mode == tf.estimator.ModeKeys.TRAIN, data_format=data_format,
            vis=vis, prefix="encoder")

        # output size is vocab size + 1 for the blank symbol in CTC
        # note: train has no effect if batchnorm is disabled
        logits_means, _ = conv_layer(
            inputs=pre_out, n_filters=vocab_size + 1, width_filters=1,
            stride_filters=1, dilation=1,
            act=None, batchnorm=False, train=False, data_format=data_format,
            vis=vis, name="logits")
        latent_means, _ = conv_layer(
            inputs=pre_out, n_filters=bottleneck, width_filters=1,
            stride_filters=1, dilation=1,
            act=None, batchnorm=False, train=False, data_format=data_format,
            vis=vis, name="latent")
        if random:
            latent_logvar, _ = conv_layer(
                inputs=pre_out, n_filters=bottleneck, width_filters=1,
                stride_filters=1, dilation=1,
                act=None, batchnorm=False, train=False,
                data_format=data_format, vis=vis, name="latent_logvar")
            latent_samples = tf.random_normal(tf.shape(latent_means))
            latent = latent_means + (latent_samples *
                                     tf.sqrt(tf.exp(latent_logvar)))
            if full_vae:
                logits_logvar, _ = conv_layer(
                    inputs=pre_out, n_filters=vocab_size + 1, width_filters=1,
                    stride_filters=1, dilation=1,
                    act=None, batchnorm=False, train=False,
                    data_format=data_format, vis=vis, name="logits_logvar")
                logits_samples = tf.random_normal(tf.shape(logits_means))
                logits = logits_means + (logits_samples *
                                         tf.sqrt(tf.exp(logits_logvar)))
            else:
                logits = logits_means
        else:
            logits = logits_means
            latent = latent_means

        if only_decode:
            joint = features["latent"]
            if cf:
                logits = joint[:, :(vocab_size+1), :]
                latent = joint[:, (vocab_size+1):, :]
            else:
                logits = joint[:, :, :(vocab_size+1)]
                latent = joint[:, :, (vocab_size+1):]

        if topk:
            if cf:
                logits_cl = tf.transpose(logits, [0, 2, 1])
            else:
                logits_cl = logits
            _, k_inds = tf.nn.top_k(logits_cl, k=topk, sorted=False)

            # TODO: this might be an inefficient way to get a "k-hot" vector...
            char_identities = tf.one_hot(k_inds[:, :, 0], depth=vocab_size + 1)
            for ii in range(1, topk):
                char_identities += tf.one_hot(k_inds[:, :, ii],
                                              depth=vocab_size + 1)
            if cf:
                char_identities = tf.transpose(char_identities, [0, 2, 1])
            # TODO: maybe we need to embed character identities?
        else:
            char_identities = logits

        joint = tf.concat([char_identities, latent], axis=1 if cf else 2)

        pre_rec, _, decoder_layers = read_apply_model_config(
            model_config + "_dec", joint, act=act, batchnorm=use_bn,
            train=mode == tf.estimator.ModeKeys.TRAIN, data_format=data_format,
            vis=vis and ae_coeff, prefix="decoder")
        reconstructed, _ = transposed_conv_layer(
            pre_rec, n_channels, 1, 1, 1,
            act=None, batchnorm=False, train=False, data_format=data_format,
            vis=vis and ae_coeff, name="reconstructed")

    # after this we need logits in shape time x batch_size x vocab_size
    # bs x v x t -> t x bs x v    if cf, or    bs x t x v -> t x bs x v
    logits_tm = tf.transpose(logits, perm=[2, 0, 1] if cf else [1, 0, 2],
                             name="logits_time_major")

    # we need the "actual" input length *after* strided convolutions for CTC
    seq_lengths_original = seq_lengths  # to save them in predictions
    if total_stride > 1:
        seq_lengths = tf.cast(seq_lengths / total_stride, tf.int32)

    # losses come before predict because we need the CTC loss to "predict"
    # adversarial gradients
    with tf.name_scope("loss"):
        total_loss = 0
        if use_ctc:
            print("Building CTC loss...")
            # ctc wants the labels as a sparse tensor
            # note that labels are NOT time-major, but this is intended
            with tf.name_scope("ctc"):
                labels_sparse = dense_to_sparse(labels, sparse_val=-1)
                ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(
                    labels=labels_sparse, inputs=logits_tm,
                    sequence_length=seq_lengths, time_major=True),
                    name="avg_loss")
            tf.summary.scalar("ctc_loss", ctc_loss)
            total_loss += ctc_loss

            if blank_coeff or verbose_losses:
                blank_avg = blank_prob_loss(logits_tm)
                total_loss += blank_coeff * blank_avg

            # TODO doesn't mix with other losses/regularizers etc!
            # use with pure CTC!
            if adversarial:
                adversarial_gradients = tf.gradients(ctc_loss, audio)[0]
                scaled_grads = adversarial_gradients / tf.reduce_max(
                    tf.abs(adversarial_gradients), axis=[1, 2], keepdims=True)
                adv_audio = audio + 20 * scaled_grads
                adv_audio = tf.clip_by_value(adv_audio, np.log(1e-10),
                                             tf.reduce_max(audio))
                adv_audio = tf.stop_gradient(adv_audio)

                with tf.variable_scope("model", reuse=True):
                    adv_pre_out, adv_total_stride, adv_encoder_layers = \
                        read_apply_model_config(
                            model_config + "_enc", adv_audio, act=act,
                            batchnorm=use_bn,
                            train=mode == tf.estimator.ModeKeys.TRAIN,
                            data_format=data_format, vis=vis, prefix="encoder")

                    # output size is vocab size + 1 for the blank symbol in CTC
                    # note: train has no effect if batchnorm is disabled
                    adv_logits, _ = conv_layer(
                        adv_pre_out, vocab_size + 1, 1, 1, 1,
                        act=None, batchnorm=False, train=False,
                        data_format=data_format,
                        vis=vis, name="logits")
                    adv_logits_tm = tf.transpose(
                        adv_logits, perm=[2, 0, 1] if cf else [1, 0, 2],
                        name="logits_time_major")

                adv_ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(
                    labels=labels_sparse, inputs=adv_logits_tm,
                    sequence_length=seq_lengths, time_major=True),
                    name="avg_loss")
                tf.summary.scalar("adv_ctc_loss", adv_ctc_loss)
                total_loss += adv_ctc_loss
                total_loss /= 2

        if ae_coeff:
            print("Building reconstruction loss...")
            reconstr_loss = reconstruction_loss(
                audio, reconstructed, seq_lengths_original, phase, n_channels,
                phase_freqs_in_data, cf)
            ae_loss = reconstr_loss

            if mmd_coeff or verbose_losses:
                print("Building MMD loss...")
                mask_latent = tf.sequence_mask(seq_lengths)
                mmd = mmd_loss(latent, logits, mask_latent, full_vae, cf)
                ae_loss += mmd_coeff * mmd

            if random:
                print("Building variance loss for random encoder...")
                if full_vae:
                    latent_logvar = tf.concat([latent_logvar, logits_logvar],
                                              axis=1 if cf else -1)
                enc_var_loss = tf.reduce_mean(tf.abs(latent_logvar))
                tf.summary.scalar("enc_var_loss", enc_var_loss)
                ae_loss += random * enc_var_loss

            if reg_coeff or verbose_losses:
                print("Building latent variance loss...")
                # we assume channels_first in the regularizer
                if not cf:
                    latent_cf = tf.transpose(latent, [0, 2, 1])
                else:
                    latent_cf = latent
                latent_var_loss = feature_map_local_variance_regularizer(
                    latent_cf, "cos", mask_latent)
                tf.summary.scalar("latent_var_loss", latent_var_loss)
                ae_loss += reg_coeff * latent_var_loss

            total_loss += ae_coeff * ae_loss
            # TODO: maybe use non-random values for regularizer and logits

        if l2_reg or verbose_losses:
            l2_loss = tf.losses.get_regularization_loss()
            tf.summary.scalar("kernel_l2_loss", l2_loss)
            total_loss += l2_reg * l2_loss

    if mode == tf.estimator.ModeKeys.PREDICT:
        with tf.name_scope("predictions"):
            # predictions have to map strings to tensors, so I can't just
            # add the encoder/decoder layer lists -- these are repackaged in
            # estimator_main.py
            predictions = {"logits": logits,
                           "probabilities": tf.nn.softmax(
                               logits,
                               dim=1 if cf else -1,
                               name="softmax_probabilities"),
                           "latent": latent,
                           "input": audio,
                           "input_length": seq_lengths_original,
                           "reconstruction": reconstructed}
            if random:
                predictions["latent_means"] = latent_means
                predictions["latent_logvar"] = latent_logvar
                if full_vae:
                    predictions["logits_means"] = logits_means
                    predictions["logits_logvar"] = logits_logvar

            for name, act in encoder_layers + decoder_layers:
                predictions[name] = act
            if use_ctc:
                decoded = decode(logits_tm, seq_lengths, top_paths=100,
                                 pad_val=-1)
                predictions["decoding"] = decoded
                if adversarial:
                    predictions["adversarial_gradient"] = tf.gradients(
                        ctc_loss, audio)[0]
            if not phase:
                predictions["audio_with_phase"] = audio_with_phase

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.variable_scope("optimizer"):
            print("Building optimizer...")
            loss_history = tf.Variable(np.zeros(20000), trainable=False,
                                       dtype=tf.float32, name="loss_history")
            lr = tf.Variable(adam_args[0], trainable=False, dtype=tf.float32,
                             name="learning_rate")
            tf.summary.scalar("learning_rate", lr)

            update_lh = tf.assign(loss_history,
                                  tf.concat((loss_history[1:], [total_loss]),
                                            axis=0),
                                  name="update_loss_history")
            update_lr = tf.assign(
                lr, lr_annealer(
                    lr, 0.1, update_lh, tf.train.get_global_step()),
                name="update_learning_rate")

            if fix_lr:
                lr_used = adam_args[0]
            else:
                lr_used = update_lr

            if momentum:
                optimizer = tf.train.MomentumOptimizer(
                    lr_used, adam_args[1], use_nesterov=True)
            else:
                optimizer = tf.train.AdamOptimizer(lr_used, *adam_args[1:])
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
            if use_bn:
                # necessary for batchnorm to work properly in inference mode
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op, grads_and_vars, glob_grad_norm = clip_and_step(
                        optimizer, total_loss, clipping)
            else:
                train_op, grads_and_vars, glob_grad_norm = clip_and_step(
                    optimizer, total_loss, clipping)
        # visualize gradients
        if vis:
            with tf.name_scope("gradient_norm"):
                for g, v in grads_and_vars:
                    if v.name.find("kernel") >= 0 and g is not None:
                        tf.summary.scalar(v.name + "gradient_norm", tf.norm(g))
                tf.summary.scalar("global_gradient_norm", glob_grad_norm)

        # The combined summary/profiling hook needs to be created in here
        scaff = tf.train.Scaffold()
        hooks = []
        if vis:
            save_and_profile = SummarySaverHookWithProfile(
                save_steps=vis, profile_steps=50*vis,
                output_dir=config.model_dir, scaffold=scaff)
            hooks.append(save_and_profile)
        return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss,
                                          train_op=train_op, scaffold=scaff,
                                          training_hooks=hooks)

    # if we made it here, we are in evaluation mode
    # NOTE that this has only letter error rate; word error rates can be
    # obtained from run_asr in "errors" mode.
    # errors mode is preferred for LER as well due to proper micro-averaging
    with tf.name_scope("evaluation"):
        eval_metric_ops = {}
        if ae_coeff:
            eval_metric_ops["reconstruction_loss"] = tf.metrics.mean(
                reconstr_loss, name="reconstruction_eval")
            if mmd_coeff or verbose_losses:
                eval_metric_ops["mmd_loss"] = tf.metrics.mean(
                    mmd, name="mmd_eval")
            if reg_coeff or verbose_losses:
                eval_metric_ops["latent_var_loss"] = tf.metrics.mean(
                    latent_var_loss, name="latent_var_eval")
            if random:
                eval_metric_ops["enc_var_loss"] = tf.metrics.mean(
                    enc_var_loss, name="enc_var_eval")
        if use_ctc:
            decoded = decode_top(logits_tm, seq_lengths, pad_val=-1,
                                 as_sparse=True)
            ed_dist = tf.reduce_mean(tf.edit_distance(decoded, labels_sparse),
                                     name="edit_distance_batch_mean")
            eval_metric_ops["edit_distance"] = tf.metrics.mean(
                ed_dist, name="edit_distance_eval")
            eval_metric_ops["ctc_loss"] = tf.metrics.mean(
                ctc_loss, name="ctc_eval")
            if blank_coeff or verbose_losses:
                eval_metric_ops["blank_activity"] = tf.metrics.mean(
                    blank_avg, name="blank_activity")
    return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss,
                                      eval_metric_ops=eval_metric_ops)


###############################################################################
# Helper functions for building inference models.
###############################################################################
def read_apply_model_config(config_path, inputs, act, batchnorm, train,
                            data_format, vis, prefix):
    """Read a model config file and apply it to an input.

    A config file is a csv file where the header determines the network type.
    The header should follow the format: model_type where model type is one of
    cnn, cnn_t (transposed), lstm.
    After that, each line stands for a layer or a whole residual block.

    If using CNN, lines should follow the format: type,n_f,w_f,s_f,d_f
        type: "layer", "block" or "dense", stating whether this is a single
              conv layer, a residual block or a dense block.
        n_f: Number of filters in the layer/block. For dense blocks, this is
             the growth rate!
        w_f: Width of filters.
        s_f: Convolutional stride of the layer/block.
        d_f: Dilation rate of the layer/block.

    If using RNN, lines should follow the format: size
        size: Size of the RNN hidden layer.

    Parameters:
        config_path: Path to the model config file.
        inputs: 3D inputs to the model (audio).
        act: Activation function to apply in each layer/block.
        batchnorm: Bool, whether to use batch normalization.
        train: Bool or TF placeholder. Fed straight into batch normalization
               (ignored if that is not used).
        data_format: channels_first or _last. Assumed that you checked validity
                     beforehand. I.e. if it's not first, this function simply
                     assumes that it's last.
        vis: Bool, whether to add histograms for layer activations.
        prefix: String to prepend to all layer names (also creates variable
                scope).

    Returns:
        Output of the last layer/block, total stride of the network and a list
        of all layer/block activations with their names (tuples name, act).

    """
    print("Reading, building and applying {}...".format(prefix))
    with open(config_path) as model_config:
        config_strings = model_config.readlines()
        model_type = config_strings[0].strip()
        if model_type.split("_")[0] == "cnn":
            transpose = model_type == "cnn_t"
            parsed_config = [parse_model_config_line_cnn(line) for
                             line in config_strings[1:]]

            output, total_stride, all_layers, total_pars = cnn1d_from_config(
                parsed_config, inputs, act, batchnorm, train, data_format, vis,
                transpose, prefix)
        elif model_type == "lstm":
            parsed_config = [parse_model_config_line_rnn(line) for
                             line in config_strings[1:]]
            output, total_stride, all_layers, total_pars = rnn_from_config(
                parsed_config, inputs, data_format, vis, prefix)
        else:
            raise ValueError("Invalid model type {} "
                             "specified.".format(model_type))

    print("Number of model parameters ({}): {}".format(prefix, total_pars))
    return output, total_stride, all_layers


def parse_model_config_line_cnn(l):
    """Parse a single config line for CNN models."""
    entries = l.strip().split(",")
    return (entries[0], int(entries[1]), int(entries[2]), int(entries[3]),
            int(entries[4]))


def parse_model_config_line_rnn(l):
    """Parse a single config line for RNN models."""
    entries = l.strip().split(",")
    return (int(entries[0]),)
