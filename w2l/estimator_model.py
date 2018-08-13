import numpy as np
import tensorflow as tf

from .utils.hooks import SummarySaverHookWithProfile
from .utils.layers import (conv_layer, residual_block, dense_block,
                           transposed_conv_layer)
from .utils.model import (decode, decode_top, dense_to_sparse, lr_annealer,
                          clip_and_step, compute_mmd,
                          feature_map_global_variance_regularizer,
                          feature_map_local_variance_regularizer)


def w2l_model_fn(features, labels, mode, params, config):
    """Model function for tf.estimator.
    
    Parameters:
        features: Should be a dict containing string keys:
            audio: batch_size x channels x seq_len tensor of input sequences.
                   Note: Must be channels_first!!
            length: batch_size tensor of sequence lengths for each input.
        labels: Should be a dict containing string keys:
            transcription: batch_size x label_len tensor of label indices
                           (standing for letters).
            length: batch_size tensor of sequence lengths for each
                    transcription.
        mode: Train, Evaluate or Predict modes from tf.estimator.
        params: Should be a dict with the following string keys:
            model_config: Path to config file to build the model up to the
                          pre-final layer..
            vocab_size: Size of the vocabulary, to get the size for the final
                        layer.
            act: The activation function, e.g. tf.nn.relu or tf.nn.elu.
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
            mmd: Coefficient for MMD loss for latent space (Wasserstein VAE
                 thingy).
            bottleneck: Size of bottleneck (latent representation).
            use_ctc: Bool, whether to use CTC loss.
            ae_coeff: Coefficient for AE loss.
            only_decode: Bool, if set only run the decoder and assume that the
                         inputs are logits + latent space samples.
            phase: If false, discard the phase in the input (currently hardcoded!)
            topk: Int, if > 0 only keep information on the top k logits at each
                  time step.
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
    mmd = params["mmd"]
    bottleneck = params["bottleneck"]
    use_ctc = params["use_ctc"]
    ae_coeff = params["ae_coeff"]
    only_decode = params["only_decode"]
    phase = params["phase"]
    topk = params["topk"]

    # construct model input -> output
    audio, seq_lengths = features["audio"], features["length"]
    n_channels = audio.shape.as_list()[1]
    if not phase:
        n_channels -= 201  # TODO don't hardcode
        audio_with_phase = audio
        audio = audio[:, :n_channels, :]
    if labels is not None:  # predict passes labels=None...
        labels = labels["transcription"]
    if data_format == "channels_last":
        audio = tf.transpose(audio, [0, 2, 1])

    with tf.variable_scope("model"):
        audio_normed = tf.layers.batch_normalization(
            audio, axis=1 if data_format == "channels_first" else -1,
            training=mode == tf.estimator.ModeKeys.TRAIN, name="input_norm")

        pre_out, total_stride, encoder_layers = read_apply_model_config(
            model_config, audio_normed, act=act, batchnorm=use_bn,
            train=mode == tf.estimator.ModeKeys.TRAIN, data_format=data_format,
            vis=vis)

        # output size is vocab size + 1 for the extra "trash symbol" in CTC
        logits, _ = conv_layer(
            pre_out, vocab_size + 1, 1, 1, 1,
            act=None, batchnorm=False, train=False, data_format=data_format,
            vis=vis, name="logits")
        latent, _ = conv_layer(
            pre_out, bottleneck, 1, 1, 1,
            act=None, batchnorm=False, train=False, data_format=data_format,
            vis=vis, name="latent")

        if only_decode:
            joint = features["latent"]
            if data_format == "channels_first":
                logits = joint[:, :29, :]
                latent = joint[:, 29:, :]
            else:
                logits = joint[:, :, :29]
                latent = joint[:, :, 29:]

        if topk:
            if data_format == "channels_first":
                logits_cl = tf.transpose(logits, [0, 2, 1])
            else:
                logits_cl = logits
            k_vals, k_inds = tf.nn.top_k(logits_cl, k=topk, sorted=False)

            # TODO this might be an inefficient way to get a "k-hot" vector...
            char_identities = tf.one_hot(k_inds[:, :, 0], depth=vocab_size + 1)
            for ii in range(1, topk):
                char_identities += tf.one_hot(k_inds[:, :, ii], depth=vocab_size + 1)
            if data_format == "channels_first":
                char_identities = tf.transpose(char_identities, [0, 2, 1])
            # TODO maybe we need to embed character identities?
        else:
            char_identities = logits

        joint = tf.concat([char_identities, latent],
                          axis=1 if data_format == "channels_first" else 2)

        pre_rec, decoder_layers = read_apply_model_config_inverted(
            model_config, joint, act=act, batchnorm=use_bn,
            train=mode == tf.estimator.ModeKeys.TRAIN, data_format=data_format,
            vis=vis)
        reconstructed, _ = transposed_conv_layer(
            pre_rec, n_channels, 1, 1, 1,
            act=None, batchnorm=False, train=False, data_format=data_format,
            vis=vis, name="reconstructed")

    # after this we need logits in shape time x batch_size x vocab_size
    if data_format == "channels_first":  # bs x v x t -> t x bs x v
        logits_tm = tf.transpose(logits, [2, 0, 1], name="logits_time_major")
    else:  # channels last: bs x t x v -> t x bs x v
        logits_tm = tf.transpose(logits, [1, 0, 2], name="logits_time_major")

    # we need the "actual" input length *after* strided convolutions for CTC
    # TODO the correctness of this needs to be verified
    seq_lengths_original = seq_lengths  # to save them in predictions
    if total_stride > 1:
        seq_lengths = tf.cast(seq_lengths / total_stride, tf.int32)

    if mode == tf.estimator.ModeKeys.PREDICT:
        with tf.name_scope("predictions"):
            # note that "all_layers" does not include the outputs (logits)
            # predictions have to map strings to tensors, so I can't just
            # add the list -- beware the loss of ordering in dict
            predictions = {"logits": logits,
                           "probabilities": tf.nn.softmax(
                               logits,
                               dim=1 if data_format == "channels_first" else -1,
                               name="softmax_probabilities"),
                           "latent": latent,
                           "input": audio,
                           "input_length": seq_lengths_original,
                           "reconstruction": reconstructed}
            for name, act in encoder_layers + decoder_layers:
                predictions[name] = act
            if use_ctc:
                decoded = decode(logits_tm, seq_lengths, top_paths=100,
                                 pad_val=-1)
                predictions["decoding"] = decoded
            if not phase:
                predictions["audio_with_phase"] = audio_with_phase

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    with tf.name_scope("loss"):
        total_loss = 0
        if use_ctc:
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
        if ae_coeff:
            with tf.name_scope("reconstruction_loss"):
                mask = tf.sequence_mask(seq_lengths_original, dtype=tf.float32)
                if data_format == "channels_first":
                    mask = mask[:, tf.newaxis, :]
                else:
                    mask = mask[:, :, tf.newaxis]
                # since we don't count errors on padding elements we need to be
                # careful counting the total number of elements for averaging
                reconstr_loss = (tf.reduce_sum(
                    tf.squared_difference(audio, reconstructed) * mask) /
                                 (n_channels * tf.count_nonzero(
                                     mask, dtype=tf.float32)))
            tf.summary.scalar("reconstruction_loss", reconstr_loss)

            # TODO random encoder
            with tf.name_scope("mmd"):
                if data_format == "channels_first":
                    latent_cl = tf.transpose(latent, [0, 2, 1])
                else:
                    latent_cl = latent
                # we only take each 20th entry in the time axis as sample
                # this is to reduce RAM usage but should also reduce
                # dependencies between the samples (since technically they are
                # assumed to be independent, I believe...)
                latent_flat = tf.reshape(latent_cl,
                                         [-1, tf.shape(latent_cl)[-1]])[::20]
                mask_latent = tf.sequence_mask(seq_lengths)
                mask_flat = tf.reshape(mask_latent, [-1])[::20]
                latent_masked = tf.boolean_mask(latent_flat, mask_flat)
                target_samples = tf.random_normal(tf.shape(latent_masked))
                mmd_loss = compute_mmd(target_samples, latent_masked)
            ae_loss = reconstr_loss
            if mmd:
                ae_loss += mmd*mmd_loss
            tf.summary.scalar("mmd_loss", mmd_loss)
            total_loss += ae_coeff * ae_loss

            if reg_coeff:
                # we assume channels_first in the regularizer and don't need
                # latent afterwards so we transpose "in place"
                if data_format == "channels_last":
                    latent = tf.transpose(latent, [0, 2, 1])
                reg_loss = feature_map_local_variance_regularizer(
                    latent, "cos", mask_latent)
                tf.summary.scalar("reg_loss", reg_loss)
                total_loss += reg_coeff * reg_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.variable_scope("optimizer"):
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
            with tf.name_scope("visualization"):
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
    # NOTE that this is only letter error rate; word error rates can be
    # obtained from run_asr in "errors" mode.
    with tf.name_scope("evaluation"):
        eval_metric_ops = {}
        if ae_coeff:
            eval_metric_ops["reconstruction_loss"] = tf.metrics.mean(
                reconstr_loss, name="reconstruction_eval")
            eval_metric_ops["mmd_loss"] = tf.metrics.mean(
                mmd_loss, name="mmd_eval")
        if use_ctc:
            decoded = decode_top(logits_tm, seq_lengths, pad_val=-1,
                                 as_sparse=True)
            ed_dist = tf.reduce_mean(tf.edit_distance(decoded, labels_sparse),
                                     name="edit_distance_batch_mean")
            eval_metric_ops["edit_distance"] = tf.metrics.mean(
                ed_dist, name="edit_distance_eval")
            eval_metric_ops["ctc_loss"] = tf.metrics.mean(
                ctc_loss, name="ctc_eval")
    return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss,
                                      eval_metric_ops=eval_metric_ops)


###############################################################################
# Helper functions for building inference models.
###############################################################################
def read_apply_model_config(config_path, inputs, act, batchnorm, train,
                            data_format, vis):
    """Read a model config file and apply it to an input.

    A config file is a csv file where each line stands for a layer or a whole
    residual block. Lines should follow the format:
    type,n_f,w_f,s_f,d_f
        type: "layer", "block" or "dense", stating whether this is a single
              conv layer, a residual block or a dense block.
        n_f: Number of filters in the layer/block. For dense blocks, this is
             the growth rate!
        w_f: Width of filters.
        s_f: Convolutional stride of the layer/block.
        d_f: Dilation rate of the layer/block.

    NOTE: This is for 1D convolutional models without pooling like Wav2letter.
          The final layer should *not* be included since it's always the same
          and depends on the data (i.e. vocabulary size).

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

    Returns:
        Output of the last layer/block, total stride of the network and a list
        of all layer/block activations with their names (tuples name, act).
    """
    # TODO for resnets/dense nets, return all *layers*, not just blocks
    print("Reading, building and applying encoder...")
    total_pars = 0
    all_layers = []
    with open(config_path) as model_config:
        total_stride = 1
        previous = inputs
        config_strings = model_config.readlines()

        for ind, line in enumerate(config_strings):
            t, n_f, w_f, s_f, d_f = parse_model_config_line(line)
            name = "encoder_" + t + str(ind)
            if t == "layer":
                previous, pars = conv_layer(
                    previous, n_f, w_f, s_f, d_f, act, batchnorm, train,
                    data_format, vis, name=name)
            # TODO residual/dense blocks ignore some parameters ATM!
            elif t == "block":
                previous, pars = residual_block(
                    previous, n_f, w_f, s_f, act, batchnorm, train,
                    data_format, vis, name=name)
            elif t == "dense":
                previous, pars = dense_block(
                    previous, n_f, w_f, s_f, act, batchnorm, train,
                    data_format, vis, name=name)
            else:
                raise ValueError(
                    "Invalid layer type specified in layer {}! Valid are "
                    "'layer', 'block', 'dense'. You specified "
                    "{}.".format(ind, t))
            all_layers.append((name, previous))
            total_stride *= s_f
            total_pars += pars
    print("Number of model parameters (encoder): {}".format(total_pars))
    return previous, total_stride, all_layers


def read_apply_model_config_inverted(config_path, inputs, act, batchnorm,
                                     train, data_format, vis):
    """As above, but applies the config in reverse order with transposed
     convolutions.

     Note: The last layer of the encoder is not applied in transposed fashion.
           An additional layer is added to get back to the input dimensionality

     Parameters:
        config_path: Path to the model config file.
        inputs: 3D inputs to the model (pre-logits layer of "encoder").
        act: Activation function to apply in each layer/block.
        batchnorm: Bool, whether to use batch normalization.
        train: Bool or TF placeholder. Fed straight into batch normalization
               (ignored if that is not used).
        data_format: channels_first or _last. Assumed that you checked validity
                     beforehand. I.e. if it's not first, this function simply
                     assumes that it's last.
        vis: Bool, whether to add histograms for layer activations.

    Returns:
        Output of the last layer/block and a list of all layer/block
        activations with their names (tuples name, act).

    """
    # TODO refactor with the above function
    print("Reading, building and applying decoder...")
    total_pars = 0
    all_layers = []
    with open(config_path) as model_config:
        total_stride = 1
        previous = inputs
        config_strings = model_config.readlines()

        for ind, line in enumerate(reversed(config_strings)):
            # TODO this sucks lol
            t, n_f, w_f, s_f, d_f = parse_model_config_line(line)
            try:
                n_f = int(parse_model_config_line(config_strings[ind + 1])[1])
            except:
                n_f = 256

            name = "decoder_" + t + str(ind)
            if t == "layer":
                previous, pars = transposed_conv_layer(
                    previous, n_f, w_f, s_f, d_f, act, batchnorm, train,
                    data_format, vis, name=name)
            # TODO residual/dense blocks ignore some parameters ATM!
            elif t == "block":
                previous, pars = residual_block(
                    previous, n_f, w_f, s_f, act, batchnorm, train,
                    data_format, vis, name=name)
            elif t == "dense":
                previous, pars = dense_block(
                    previous, n_f, w_f, s_f, act, batchnorm, train,
                    data_format, vis, name=name)
            else:
                raise ValueError(
                    "Invalid layer type specified in layer {}! Valid are "
                    "'layer', 'block', 'dense'. You specified "
                    "{}.".format(ind, t))
            all_layers.append((name, previous))
            total_stride *= s_f
            total_pars += pars
    print("Number of model parameters (decoder): {}".format(total_pars))
    return previous, all_layers


def parse_model_config_line(l):
    entries = l.strip().split(",")
    return (entries[0], int(entries[1]), int(entries[2]), int(entries[3]),
            int(entries[4]))
