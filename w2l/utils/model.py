import numpy as np
import tensorflow as tf


###############################################################################
# Regularizer
###############################################################################
def feature_map_global_variance_regularizer(feature_map):
    """Compute mean variance for a batch of 1D conv feature maps.

    Parameters:
        feature_map: 3D tensor of shape batch x time x channels. Note that this
        is always assumed to be channels_first! Do the transformation
        beforehand.

    Returns: Mean variance over the time axis.
    """
    with tf.name_scope("global_variance_regularizer"):
        _, var = tf.nn.moments(feature_map, axes=2)
        return tf.reduce_mean(var)


def feature_map_local_variance_regularizer(feature_map, diff_norm,
                                           neighbor_size):
    # TODO as of now, this loss is NOT INVARIANT TO BATCH SIZE!!
    """Creates a neighborhood distance regularizer.
    Parameters:
        feature_map: 3D tensor of shape batch x channels x time. Note that this
        is always assumed to be channels_first! Do the transformation
        beforehand if necessary.
        diff_norm: How to compute differences/distances between filters.
                   Can be "l1", "l2" or "linf" for respective norms, or "cos"
                   for cosine distance..
        grid_dims: 2-tuple or list giving the desired grid dimensions. Has to
                   match the number of filters for the layer to regularize.
        neighbor_size: int, giving the size of the neighborhood. Must be odd.
                       E.g. giving 3 here will cause each filter to treat the
                       immediately surrounding filters as its neighborhood.
    """
    if not neighbor_size % 2:
        raise ValueError("Neighborhood is not odd; this would mean no middle "
                         "point!")

    # IDEA
    # collect IDs for centers and corresponding neighbors
    # gather actual filters
    # FLATTEN batch & channel? -> time x (batch*channel), gather from there
    # shouldn't matter since we average over batch & channels anyway

    with tf.name_scope("local_variance_regularizer"):
        time = tf.shape(feature_map)[-1]
        fmap_flat = tf.reshape(feature_map, [-1, time])
        fmap_flat = tf.transpose(fmap_flat)

        neighbors_per_direction = (neighbor_size - 1) // 2
        center_ids = tf.range(neighbors_per_direction, time, dtype=tf.int32)

        neighbor_offsets = list(range(-neighbors_per_direction,
                                      neighbors_per_direction + 1))
        neighbor_offsets.pop(neighbors_per_direction)  # remove center

        def do_offset(offset):
            return tf.range(neighbors_per_direction + offset, time + offset,
                            dtype=tf.int32)
        neighbor_ids = tf.stack([do_offset(offset)
                                 for offset in neighbor_offsets],
                                axis=1)

        neighbor_offsets = np.asarray(neighbor_offsets, dtype=np.int32)

        neighbor_weights = 1. / np.abs(neighbor_offsets.astype(np.float32))
        neighbor_weights /= neighbor_weights.sum()  # normalize to sum=1
        # neighbor_weights /= np.sqrt((neighbor_weights ** 2).sum())  # normalize to length=1
        neighbor_weights = tf.constant(neighbor_weights,
                                       name='neighbor_weights')

        # broadcast to n_centers x 1 x d
        centers = tf.gather(fmap_flat, center_ids)
        centers = tf.expand_dims(centers, 1)

        # n_centers x n_neighbors x d
        neighbors = tf.gather(fmap_flat, neighbor_ids)

        # compute pairwise distances, then weight, then sum up
        # pairwise is always n_centers x n_neighbors
        if diff_norm == "l1":
            pairwise = tf.reduce_sum(tf.abs(centers - neighbors),
                                     axis=-1)
        elif diff_norm == "l2":
            pairwise = tf.sqrt(
                tf.reduce_sum((centers - neighbors) ** 2, axis=-1))
        elif diff_norm == "linf":
            pairwise = tf.reduce_max(tf.abs(centers - neighbors),
                                     axis=-1)
        elif diff_norm == "cos":
            dotprods = tf.reduce_sum(centers * neighbors, axis=-1)
            center_norms = tf.norm(centers, axis=-1)
            neighbor_norms = tf.norm(neighbors, axis=-1)
            # NOTE this computes cosine *similarity* which is why we
            # multiply by -1: minimize the negative similarity!
            cosine_similarity = dotprods / (center_norms * neighbor_norms)
            pairwise = -1 * cosine_similarity
        else:
            raise ValueError("Invalid difference norm specified: {}. "
                             "Valid are 'l1', 'l2', 'linf', "
                             "'cos'.".format(diff_norm))
        pairwise_weighted = neighbor_weights * pairwise
        return tf.reduce_sum(pairwise_weighted) / tf.shape(pairwise_weighted)[0]


###############################################################################
# Helper functions for various purposes.
###############################################################################
def clip_and_step(optimizer, loss, clipping):
    """Helper to compute/apply gradients with clipping.

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
    grads_and_vars = optimizer.compute_gradients(loss)
    grads, varis = zip(*grads_and_vars)
    if clipping:
        grads, global_norm = tf.clip_by_global_norm(grads, clipping,
                                                    name="gradient_clipping")
    else:
        global_norm = tf.global_norm(grads, name="gradient_norm")
    grads_and_vars = list(zip(grads, varis))  # list call is apparently vital!!
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
    """Helper for ctc decoding.

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
        decoded_sparse = decoded_sparse_list[0]
        # dunno if there's a better way to convert dtypes
        decoded_sparse = tf.cast(decoded_sparse, tf.int32)
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

    Parameters:
        lr: Tensor containing the current learning rate.
        factor: By what to multiply the learning rate in case of annealing.
        loss_history: Tensor containing the last n loss values. Used to judge
                      whether it's decreasing or not.
        global_step: Tensor containing the global step. As it is right now,
                     a check is only made if global step % n = 0.
    """
    n = loss_history.shape.as_list()[0]

    def reduce_if_slope():
        """Evaluated every 10k steps or so. Lowers LR if slope >= 0."""
        with tf.name_scope("regression"):
            x1 = tf.range(n, dtype=tf.float32, name="x")
            x2 = tf.ones([n], dtype=tf.float32, name="bias")
            x = tf.stack((x1, x2), axis=1, name="input")
            slope_bias = tf.matrix_solve_ls(
                x, tf.expand_dims(loss_history, -1),
                name="solution")
            slope = slope_bias[0][0]
            bias = slope_bias[1][0]
            preds = slope * x1 + bias

        with tf.name_scope("slope_prob"):
            data_var = 1 / (n - 2) * tf.reduce_sum(tf.square(loss_history - preds))
            dist_var = 12 * data_var / (n ** 3 - n)
            dist = tf.distributions.Normal(slope, tf.sqrt(dist_var),
                                           name="slope_distribution")
            prob_decreasing = dist.cdf(0., name="prob_below_zero")
            return tf.cond(tf.less_equal(prob_decreasing, 0.5),
                           true_fn=lambda: lr * factor, false_fn=lambda: lr)

    return tf.cond(tf.logical_or(
        tf.greater(tf.mod(global_step, n), 0), tf.equal(global_step, 0)),
        true_fn=lambda: lr, false_fn=reduce_if_slope)


def compute_kernel(x, y):
    """Compute pairwise similarity measure between two batches of vectors.

    Parameters:
        x: n x d tensor of floats.
        y: Like x.
    Returns:
        n x n tensor where element i, j contains similarity between element i
        in x and element j in y. Or maybe it's the other way around? Who knows.
        I actually think it's the other way. :D
    """
    x_broadcast = x[:, tf.newaxis, :]
    y_broadcast = y[tf.newaxis, :, :]
    return tf.exp(
        -tf.reduce_mean(tf.square(x_broadcast - y_broadcast), axis=2) /
        tf.cast(tf.shape(x)[1], tf.float32))


def compute_mmd(x, y, sigma_sqr=1.0):
    """Compute MMD between two batches of vectors.

    Parameters:
        x: n x d tensor of floats.
        y: Like x.
    Returns:
        Scalar MMD value.
    """
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel + y_kernel - 2 * xy_kernel)
