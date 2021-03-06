"""Model functionality that does not fit in layers (regularizers etc.)."""
import numpy as np
import tensorflow as tf


###############################################################################
# Losses
###############################################################################
def blank_prob_loss(logits_cl):
    """Compute regularization loss for average blank label activity.

    Parameters:
        logits_cl: Character logits. Note that these are assumed to be
                   channels_last!!
    """
    with tf.variable_scope("blank_prob_regularizer"):
        probs = tf.nn.softmax(logits_cl)
        blank_avg = tf.reduce_mean(probs[:, :, -1])
    tf.summary.scalar("blank_activation", blank_avg)
    return blank_avg


def reconstruction_loss(audio, reconstructed, seq_lengths, phase, n_channels,
                        phase_channels, channels_first):
    """Compute reconstruction loss for an autoencoder.

    Parameters:
        audio: The original audio.
        reconstructed: Reconstructed audio.
        seq_lengths: Actual audio sequence lengths without padding.
        phase: Bool: whether the data includes phase (uses different loss).
        n_channels: Number of channels in the data. Easy to compute in here,
                    but we compute it outside anyway so might as well pass it.
        phase_channels: How many phase channels there are in the data.
        channels_first: Bool, whether data is channels_first. If not, is
                        assumed to be channels_last.
    """
    with tf.name_scope("reconstruction_loss"):
        mask_inp = tf.sequence_mask(seq_lengths, dtype=tf.float32)
        if channels_first:
            mask_inp = mask_inp[:, tf.newaxis, :]
        else:
            mask_inp = mask_inp[:, :, tf.newaxis]
        # since we don't count errors on padding elements we need to be
        # careful counting the total number of elements for averaging
        if phase:
            mag_channels = n_channels - phase_channels
            mag_audio = audio[:, :mag_channels, :]
            mag_rec = reconstructed[:, :mag_channels, :]
            phase_audio = audio[:, mag_channels:, :]
            phase_rec = (tf.constant(np.pi, dtype=tf.float32) *
                         tf.nn.tanh(reconstructed[:, mag_channels:, :]))
            reconstr_loss_mag = tf.reduce_sum(
                tf.squared_difference(mag_audio, mag_rec) * mask_inp)
            phase_diff = phase_audio - phase_rec
            # reconstr_loss_phase = tf.reduce_sum(
            #     tf.minimum(phase_diff, 2*np.pi-phase_diff)*mask_inp)
            reconstr_loss_phase = tf.reduce_sum(
                tf.sin(tf.abs(phase_diff / 2)) * mask_inp)
            reconstr_loss = (
                    (reconstr_loss_mag + reconstr_loss_phase) /
                    (n_channels * tf.count_nonzero(mask_inp,
                                                   dtype=tf.float32)))
        else:
            reconstr_loss = (tf.reduce_sum(
                tf.squared_difference(audio,
                                      reconstructed) * mask_inp) /
                             (n_channels * tf.count_nonzero(
                                 mask_inp, dtype=tf.float32)))
    tf.summary.scalar("reconstruction_loss", reconstr_loss)
    return reconstr_loss


def mmd_loss(latent, logits, latent_mask, full_vae, cf):
    """Compute MMD loss for latent space.

    Parameters:
        latent: Latent space tensor.
        logits: Logit space tensor. If full_vae is not set you may pass a dummy
                here.
        latent_mask: Mask giving which latent elements are "real" i.e. no
                     padding.
        full_vae: Bool; if true, also compute MMD on the logits.
        cf: Bool, if true data format is assumed to be channels_first (else
            channels_last).
    """
    with tf.name_scope("mmd"):
        if full_vae:
            latent_cl = tf.concat([logits, latent],
                                  axis=1 if cf else -1)
        else:
            latent_cl = latent
        if cf:
            latent_cl = tf.transpose(latent_cl, [0, 2, 1])
        # we only take each 20th entry in the time axis as sample
        # this is to reduce RAM usage but should also reduce
        # dependencies between the samples (since technically they
        # are assumed to be independent, I believe...)
        latent_flat = tf.reshape(
            latent_cl, [-1, latent_cl.shape.as_list()[-1]])[::20]
        mask_flat = tf.reshape(latent_mask, [-1])[::20]
        latent_masked = tf.boolean_mask(latent_flat, mask_flat)
        target_samples = tf.random_normal(tf.shape(latent_masked))
        mmd = compute_mmd(target_samples, latent_masked)
    tf.summary.scalar("mmd_loss", mmd)
    return mmd


###############################################################################
# Regularizer
###############################################################################
def feature_map_global_variance_regularizer(feature_map, mask):
    """Compute mean normalized variance for a batch of 1D conv feature maps.

    Parameters:
        feature_map: 3D tensor of shape batch x channels x time. Note that this
        is always assumed to be channels_first! Do the transformation
        beforehand.
        mask: 2D tensor, batch x time.

    Returns:
        Mean variance over the time axis.

    """
    with tf.name_scope("global_variance_regularizer"):
        # we apply batch norm to the data to make it scale-invariant
        batch_means, batch_var = tf.nn.moments(feature_map, axes=[0, 2],
                                               keep_dims=True)
        batch_var += 1e-10  # pad-only steps have no variance
        feature_map = (feature_map - batch_means) / tf.sqrt(batch_var)

        # for the actual variance computation, be careful to exclude padding
        nonzeros = tf.count_nonzero(mask, axis=1, dtype=tf.float32,
                                    keepdims=True)[:, tf.newaxis, :]

        means = tf.reduce_sum(feature_map, axis=2, keepdims=True) / nonzeros

        deviations = tf.squared_difference(means, feature_map)
        masked = deviations * tf.cast(mask[:, tf.newaxis, :], tf.float32)
        var = (tf.reduce_sum(masked) /
               (tf.reduce_sum(nonzeros)*tf.cast(feature_map.shape.as_list()[1],
                                                tf.float32)))
        return var


def feature_map_local_variance_regularizer(feature_map, diff_norm, mask):
    """Create a neighborhood distance regularizer.

    Parameters:
        feature_map: 3D tensor of shape batch x channels x time. Note that this
        is always assumed to be channels_first! Do the transformation
        beforehand if necessary.
        diff_norm: How to compute differences/distances between feature maps.
                   Can be "l1", "l2" or "linf" for respective norms, or "cos"
                   for cosine distance.
        mask: 2D tensor, batch x time.

    Returns:
        Scalar local variance measure.

    """
    with tf.name_scope("local_variance_regularizer"):
        # we apply batch norm to the data to make it scale-invariant
        # but only for norm-based methods (cosine is already normalized)
        if diff_norm[0] == 'l':
            batch_means, batch_var = tf.nn.moments(feature_map, axes=[0, 2],
                                                   keep_dims=True)
            batch_var += 1e-10  # pad-only steps have no variance
            feature_map = (feature_map - batch_means) / tf.sqrt(batch_var)

        fmaps_l = feature_map[:, :, :-1]
        fmaps_r = feature_map[:, :, 1:]
        # pairwise is always batch x (time-1)
        if diff_norm == "l1":
            pairwise = tf.norm(fmaps_l - fmaps_r, ord=1,  axis=1)
        elif diff_norm == "l2":
            pairwise = tf.norm(fmaps_l - fmaps_r, ord=2, axis=1)
        elif diff_norm == "linf":
            pairwise = tf.norm(fmaps_l - fmaps_r, np.inf, axis=1)
        elif diff_norm == "cos":
            norms = tf.norm(feature_map, axis=1)
            dotprods = tf.reduce_sum(fmaps_l * fmaps_r, axis=1)
            cos_sim = dotprods / (norms[:, :-1] * norms[:, 1:])
            pairwise = 1 - cos_sim
        else:
            raise ValueError("Invalid difference norm specified: {}. Valid "
                             "are 'l1', 'l2', 'linf', "
                             "'cos'.".format(diff_norm))
        masked = pairwise * tf.cast(mask[:, :-1], tf.float32)
        nonzeros = tf.count_nonzero(mask[:, :-1], dtype=tf.float32)
        return tf.reduce_sum(masked) / nonzeros


###############################################################################
# Helper functions for various purposes.
###############################################################################
def clip_and_step(optimizer, loss, clipping):
    """Compute and apply gradients with clipping.

    Parameters:
        optimizer: Subclass of tf.train.Optimizer (e.g. GradientDescent or
                   Adam).
        loss: Scalar loss tensor.
        clipping: Threshold to use for clipping.

    Returns:
        The train op.
        List of gradient, variable tuples, where gradients have been clipped.
        Global norm before clipping.

    """
    with tf.variable_scope("clip_and_step"):
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, varis = zip(*grads_and_vars)
        if clipping:
            grads, global_norm = tf.clip_by_global_norm(
                grads, clipping, name="gradient_clipping")
        else:
            global_norm = tf.global_norm(grads, name="gradient_norm")
        grads_and_vars = list(zip(grads, varis))  # list call is necessary here
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=tf.train.get_global_step(),
            name="train_step")
        return train_op, grads_and_vars, global_norm


def dense_to_sparse(dense_tensor, sparse_val=0):
    """Inverse of tf.sparse_to_dense.

    Parameters:
        dense_tensor: The dense tensor. Duh.
        sparse_val: The value to "ignore": Occurrences of this value in the
                    dense tensor will not be represented in the sparse tensor.
                    NOTE: When/if later restoring this to a dense tensor, you
                    will probably want to choose this as the default value.
    Returns:
        SparseTensor equivalent to the dense input.

    """
    with tf.name_scope("dense_to_sparse"):
        sparse_inds = tf.where(tf.not_equal(dense_tensor, sparse_val),
                               name="sparse_inds")
        sparse_vals = tf.gather_nd(dense_tensor, sparse_inds,
                                   name="sparse_vals")
        dense_shape = tf.shape(dense_tensor, name="dense_shape",
                               out_type=tf.int64)
        return tf.SparseTensor(sparse_inds, sparse_vals, dense_shape)


def decode(logits, seq_lengths, beam_width=100, top_paths=1,
           merge_repeated=False, pad_val=0, as_sparse=False):
    """Wrap ctc decoding.

    Parameters:
        logits: Passed straight to ctc decoder.
        seq_lengths: Same.
        beam_width: Same.
        top_paths: Same.
        merge_repeated: Same.
        pad_val: Value to use to pad dense tensor. No effect if as_sparse is
                 True.
        as_sparse: If True, return results as a list of sparse tensors.

    Returns:
        Either a list of sparse tensors, or a dense tensor with the requested
        top number of top predictions.

    """
    with tf.name_scope("decoding"):
        decoded_sparse_list, _ = tf.nn.ctc_beam_search_decoder(
            logits, seq_lengths, beam_width=beam_width, top_paths=top_paths,
            merge_repeated=merge_repeated)
        dc_sparse_list_int32 = [tf.cast(decoded_sparse, tf.int32)
                                for decoded_sparse in decoded_sparse_list]
        if as_sparse:
            # this is a LIST of sparse tensors!!
            return dc_sparse_list_int32
        else:
            # first we make a list where the ith entry is the batch of ith-best
            # paths (shape bs x time)
            decoded_dense_list = [tf.sparse_to_dense(
                decoded_sparse.indices,
                decoded_sparse.dense_shape,
                decoded_sparse.values,
                default_value=pad_val,
                name="dense_decoding")
                for decoded_sparse in dc_sparse_list_int32]

            # then we need to make sure all batches have the same length :(
            lengths = [tf.shape(decoded)[-1] for decoded in decoded_dense_list]
            max_length = tf.reduce_max(lengths)
            decoded_dense_list_padded = [
                tf.pad(decoded, ((0, 0), (0, max_length - length)),
                       constant_values=pad_val)
                for decoded, length in zip(decoded_dense_list, lengths)]

            # then we can finally put it into a bs x top_paths x time tensor
            decoded_tensor = tf.stack(decoded_dense_list_padded, axis=1)
            return decoded_tensor


def decode_top(logits, seq_lengths, beam_width=100, merge_repeated=False,
               pad_val=0, as_sparse=False):
    """Simpler version of ctc decoder that only returns the top result.

    Parameters:
        logits: Passed straight to ctc decoder.
        seq_lengths: Same.
        beam_width: Same.
        merge_repeated: Same.
        pad_val: Value to use to pad dense tensor. No effect if as_sparse is
                 True.
        as_sparse: If True, return results as sparse tensor.

    Returns:
        Sparse or dense tensor with the top predictions.

    """
    with tf.name_scope("decoding"):
        decoded_sparse_list, _ = tf.nn.ctc_beam_search_decoder(
            logits, seq_lengths, beam_width=beam_width, top_paths=1,
            merge_repeated=merge_repeated)
        decoded_sparse = tf.cast(decoded_sparse_list[0], tf.int32)
        if as_sparse:
            return decoded_sparse
        else:
            # this should result in a bs x t matrix of predicted classes
            return tf.sparse_to_dense(decoded_sparse.indices,
                                      decoded_sparse.dense_shape,
                                      decoded_sparse.values,
                                      default_value=pad_val,
                                      name="dense_decoding")


def repeat(inp, times):
    """np.repeat equivalent."""
    with tf.name_scope("repeat"):
        inp = tf.reshape(inp, [-1, 1])
        inp = tf.tile(inp, [1, times])
        return tf.reshape(inp, [-1])


def lr_annealer(lr, factor, loss_history, global_step):
    """Anneal the learning rate if loss doesn't decrease anymore.

    Refer to
      http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html

    Parameters:
        lr: Tensor containing the current learning rate.
        factor: By what to multiply the learning rate in case of annealing.
        loss_history: Tensor containing the last n loss values. Used to judge
                      whether it's decreasing or not.
        global_step: Tensor containing the global step. As it is right now,
                     a check is only made if global step % n = 0.

    Returns:
        The new learning rate.

    """
    with tf.variable_scope("lr_annealer"):
        n = loss_history.shape.as_list()[0]

        def reduce_if_slope():
            # Evaluated every n steps. Lowers LR if slope probably >= 0
            with tf.name_scope("regression"):
                x1 = tf.range(n, dtype=tf.float32, name="x")
                x2 = tf.ones([n], dtype=tf.float32, name="bias")
                x = tf.stack((x1, x2), axis=1, name="input")
                slope_bias = tf.matrix_solve_ls(x, loss_history[:, tf.newaxis],
                                                name="solution")
                slope = slope_bias[0][0]
                bias = slope_bias[1][0]
                preds = slope * x1 + bias

            with tf.name_scope("slope_prob"):
                data_var = 1 / (n - 2) * tf.reduce_sum(tf.square(loss_history -
                                                                 preds))
                dist_var = 12 * data_var / (n ** 3 - n)
                dist = tf.distributions.Normal(slope, tf.sqrt(dist_var),
                                               name="slope_distribution")
                prob_decreasing = dist.cdf(0., name="prob_below_zero")
                return tf.cond(tf.less_equal(prob_decreasing, 0.5),
                               true_fn=lambda: lr * factor,
                               false_fn=lambda: lr)

        return tf.cond(tf.logical_or(
            tf.greater(tf.mod(global_step, n), 0), tf.equal(global_step, 0)),
            true_fn=lambda: lr, false_fn=reduce_if_slope)


def compute_kernel(x, y, sigma_sqr):
    """Compute pairwise similarity measure between two batches of vectors.

    Parameters:
        x: n x d tensor of floats.
        y: Like x.
        sigma_sqr: Variance for the Gaussian kernel.

    Returns:
        n x n tensor where element i, j contains similarity between element i
        in x and element j in y.

    """
    x_broadcast = x[:, tf.newaxis, :]
    y_broadcast = y[tf.newaxis, :, :]
    return tf.exp(
        -tf.reduce_mean(tf.squared_difference(x_broadcast, y_broadcast),
                        axis=2) / sigma_sqr)


def compute_mmd(x, y, sigma_sqr=None):
    """Compute MMD between two batches of vectors.

    Parameters:
        x: n x d tensor of floats.
        y: Like x.
        sigma_sqr: Variance for the Gaussian kernel.

    Returns:
        Scalar MMD value.

    """
    if sigma_sqr is None:
        sigma_sqr = tf.cast(x.shape.as_list()[1], tf.float32)
    x_kernel = compute_kernel(x, x, sigma_sqr)
    y_kernel = compute_kernel(y, y, sigma_sqr)
    xy_kernel = compute_kernel(x, y, sigma_sqr)
    return tf.reduce_mean(x_kernel + y_kernel - 2 * xy_kernel)


def leaky_random(vec_size, n_samples, std=1, diffusion=0.9):
    """Return a leaky sequence of random Gaussian vectors.

    Note that this tends to converge to 0, especially for large diffusion
    values.
    This uses numpy, not TF!!!

    Parameters:
        vec_size: Size of each sample.
        n_samples: How many samples to take.
        std: Standard deviation for the Gaussian distribution
        diffusion: The 'retention factor' of the leakage.

    Returns:
        vec_size x n_samples matrix (i.e. channels_first)

    """
    samples = [std*np.random.randn(vec_size)]
    for _ in range(n_samples - 1):
        samples.append(diffusion*samples[-1] +
                       (1-diffusion)*std*np.random.randn(vec_size))
    return np.stack(samples, axis=1)
