import numpy as np
import tensorflow as tf


# IMPORTANT these dimensions are (x-axis, y-axis). This is **reversed** from
# the usual numpy (rows, columns). So if you later want to e.g. visualize
# filter grids you will need to .reshape() to the transposed dimensions.
GRIDS = {16: (4, 4), 32: (8, 4), 64: (8, 8), 128: (16, 8), 256: (16, 16),
         512: (32, 16), 1024: (32, 32), 2048: (64, 32)}


def conv_layer(inputs, n_filters, width_filters, stride_filters, dilation, act,
               batchnorm, train, data_format, vis, reg, name):
    """Build and apply a 1D convolutional layer without pooling.

    Parameters:
        inputs: 3D inputs to the layer.
        n_filters: Number of filters for the layer.
        width_filters: Filter width for the layer.
        stride_filters: Stride for the layer.
        dilation: Dilation rate of the layer.
        act: Activation function to apply after convolution (or optionally
             batch normalization).
        batchnorm: Bool, whether to use batch normalization.
        train: Bool or TF placeholder. Fed straight into batch normalization
               (ignored if that is not used).
        data_format: channels_first or _last. Assumed that you checked validity
                     beforehand. I.e. if it's not first, this function simply
                     assumes that it's last.
        vis: Bool, whether to add a histogram for layer activations.
        reg: Regularizer type to use, or None for no regularizer.
        name: Name of the layer (used for variable scope and summary).

    Returns:
        Output of the layer and number of parameters.
    """
    channel_axis = 1 if data_format == "channels_first" else -1
    n_pars = int(inputs.shape[channel_axis]) * n_filters * width_filters
    if batchnorm:  # add per-filter beta and gamma
        n_pars += 2 * n_filters
    print("\tCreating layer {} with {} parameters...".format(name, n_pars))

    with tf.variable_scope(name):
        conv = tf.layers.conv1d(
            inputs, filters=n_filters, kernel_size=width_filters,
            strides=stride_filters, dilation_rate=dilation,
            activation=None if batchnorm else act,
            use_bias=not batchnorm, padding="same", data_format=data_format,
            kernel_regularizer=sebastians_magic_trick(
                diff_norm=reg, weight_norm="l2", grid_dims=GRIDS[n_filters],
                neighbor_size=3) if reg else None,
            name="conv")

        if batchnorm:
            conv = tf.layers.batch_normalization(
                conv, axis=channel_axis, training=train, name="batch_norm")
            if act:
                conv = act(conv)
        if vis:
            tf.summary.histogram("activations_" + name,
                                 conv)
        return conv, n_pars


def transposed_conv_layer(inputs, n_filters, width_filters, stride_filters,
                          dilation, act, batchnorm, train, data_format, vis,
                          reg, name):
    """Build and apply a 1D transposed convolutional layer.

    Parameters:
        inputs: 3D inputs to the layer.
        n_filters: Number of filters for the layer.
        width_filters: Filter width for the layer.
        stride_filters: Stride for the layer.
        dilation: Ignored.
        act: Activation function to apply after convolution (or optionally
             batch normalization).
        batchnorm: Bool, whether to use batch normalization.
        train: Bool or TF placeholder. Fed straight into batch normalization
               (ignored if that is not used).
        data_format: channels_first or _last. Assumed that you checked validity
                     beforehand. I.e. if it's not first, this function simply
                     assumes that it's last.
        vis: Bool, whether to add a histogram for layer activations.
        reg: Regularizer type to use, or None for no regularizer.
        name: Name of the layer (used for variable scope and summary).

    Returns:
        Output of the layer and number of parameters.
    """
    channel_axis = 1 if data_format == "channels_first" else -1
    n_pars = int(inputs.shape[channel_axis]) * n_filters * width_filters
    if batchnorm:  # add per-filter beta and gamma
        n_pars += 2 * n_filters
    print("\tCreating layer {} with {} parameters...".format(name, n_pars))

    with tf.variable_scope(name):
        # for some godforsaken reason there is no 1d transposed conv layer??
        # b x c x t -> b x c x t x 1
        # b x t x c -> b x t x 1 x c
        inp_2d = tf.expand_dims(
            inputs, axis=3 if data_format == "channels_first" else 2)
        conv = tf.layers.conv2d_transpose(
            inp_2d, filters=n_filters, kernel_size=(width_filters, 1),
            strides=stride_filters, activation=None if batchnorm else act,
            use_bias=not batchnorm, padding="same", data_format=data_format,
            kernel_regularizer=sebastians_magic_trick(
                diff_norm=reg, weight_norm="l2", grid_dims=GRIDS[n_filters],
                neighbor_size=3) if reg else None,
            name="conv")
        if data_format == "channels_first":
            shape = [-1, n_filters, tf.shape(inputs)[-1]]
        else:
            shape = [-1, tf.shape(inputs)[1], n_filters]
        conv = tf.reshape(conv, shape)

        if batchnorm:
            conv = tf.layers.batch_normalization(
                conv, axis=channel_axis, training=train, name="batch_norm")
            if act:
                conv = act(conv)
        if vis:
            tf.summary.histogram("activations_" + name,
                                 conv)
        return conv, n_pars


def gated_conv_layer(inputs, n_filters, width_filters, stride_filters,
                     batchnorm, train, data_format, vis, name):
    """Build a GatedConv layer.

    Basically two parallel convolutions, one with linear activation and one
    with sigmoid."""
    with tf.variable_scope(name):
        out_conv, pars1 = conv_layer(
            inputs, n_filters, width_filters, stride_filters, None, batchnorm,
            train, data_format, vis, None, "out_conv")
        gate_conv, pars2 = conv_layer(
            inputs, n_filters, width_filters, stride_filters, tf.nn.sigmoid,
            batchnorm, train, data_format, vis, None, "gate_conv")
        return out_conv * gate_conv, pars1 + pars2


def residual_block(inputs, n_filters, width_filters, stride_filters, act,
                   batchnorm, train, data_format, vis, name):
    """Simple residual block variation.

    No projection implemented! This means the inputs need to have the same
    dimensionality as the outputs of this layer. So: Number of filters needs
    to be carefully chosen, and strides > 1 are not allowed for a block.

    Parameters:
        See conv_layer. Number of filters etc. are used for both layers within
        the block.

    Returns:
        Output of the block and number of parameters.
    """
    # TODO either allow projections or raise in case of incompatible filters
    print("\tCreating residual block {}...".format(name))
    if stride_filters > 1:
        raise ValueError("Strides != 1 currently not allowed for residual "
                         "blocks.")

    with tf.variable_scope(name):
        conv1, pars1 = conv_layer(
            inputs, n_filters, width_filters, stride_filters, act, batchnorm,
            train, data_format, vis, None, "conv1")
        conv2, pars2 = conv_layer(
            conv1, n_filters, width_filters, stride_filters, None, batchnorm,
            train, data_format, vis, None, "conv2")
        out = act(conv2 + inputs) if act else conv2 + inputs
        return out, pars1 + pars2


###############################################################################
# Experimental stuff that is not quite ready yet
###############################################################################
def dense_block(inputs, n_filters, width_filters, stride_filters, act,
                batchnorm, train, data_format, vis, name):
    """For building DenseNets. Not finished!!"""
    print("\tCreating dense block {}...".format(name))
    if stride_filters > 1:
        raise ValueError("Strides != 1 currently not allowed for dense "
                         "blocks.")
    channel_axis = 1 if data_format == "channels_first" else -1
    total_pars = 0

    with tf.variable_scope(name):
        for ind in range(16):  # TODO don't hardcode block size
            conv, pars = conv_layer(
                inputs, n_filters, width_filters, stride_filters, act,
                batchnorm, train, data_format, vis, None, "conv" + str(ind))
            inputs = tf.concat([inputs, conv], axis=channel_axis)
            total_pars += pars
        return inputs, total_pars


def residual_block_new(inputs, n_filters, width_filters, stride_filters, act,
                       batchnorm, train, data_format, vis, name):
    """From "Identity Mappings in Deep Residual Networks".

    No projection implemented! This means the inputs need to have the same
    dimensionality as the outputs of this layer. So: Number of filters needs
    to be carefully chosen, and strides > 1 are not allowed for a block.

    NOTE it is unclear whether this "plays nice" with non-residual blocks
    occurring before/after. Maybe don't use right now.

    Parameters:
        See conv_layer. Number of filters etc. are used for both layers within
        the block.

    Returns:
        Output of the block.
    """
    def shifted_conv_layer(_inputs, _n_filters, _width_filters,
                           _stride_filters, _act, _batchnorm, _train,
                           _data_format, _vis, _name):
        with tf.variable_scope(_name):
            conv = _inputs
            if _batchnorm:
                conv = tf.layers.batch_normalization(
                    conv, axis=1 if _data_format == "channels_first" else -1,
                    training=_train, name="batch_norm")
            if _act:
                conv = _act(conv)
            conv = tf.layers.conv1d(
                conv, filters=_n_filters, kernel_size=_width_filters,
                strides=_stride_filters,
                activation=None if _batchnorm else _act,
                use_bias=not _batchnorm, padding="same",
                data_format=_data_format,
                name="conv")
            if _vis:
                tf.summary.histogram("activations_" + _name,
                                     conv)
            return conv

    if stride_filters > 1:
        raise ValueError("Strides != 1 currently not allowed for residual "
                         "blocks.")
    with tf.variable_scope(name):
        conv1 = shifted_conv_layer(inputs, n_filters, width_filters,
                                   stride_filters, act, batchnorm, train,
                                   data_format, vis, "conv1")
        conv2 = shifted_conv_layer(conv1, n_filters, width_filters,
                                   stride_filters, None, batchnorm, train,
                                   data_format, vis, "conv2")
        return conv2 + inputs


###############################################################################
# Regularizer
###############################################################################
def sebastians_magic_trick(diff_norm, weight_norm, grid_dims, neighbor_size):
    """Creates a neighborhood distance regularizer.

    Parameters:
        diff_norm: How to compute differences/distances between filters.
                   Can be "l1", "l2" or "linf" for respective norms, or "cos"
                   for cosine distance..
        weight_norm: How to compute neighborhood weightings, i.e. how points
                     further away in the neighborhood play into the overall
                     penalty. Options same as for diff_norm, except for "cos".
        grid_dims: 2-tuple or list giving the desired grid dimensions. Has to
                   match the number of filters for the layer to regularize.
        neighbor_size: int, giving the size of the neighborhood. Must be odd.
                       E.g. giving 3 here will cause each filter to treat the
                       immediately surrounding filters (including diagonally)
                       as its neighborhood.
    """
    if not neighbor_size % 2:
        raise ValueError("Neighborhood is not odd; this would mean no middle "
                         "point!")

    # first we compute the possible offsets around a given point
    neighbors_per_direction = (neighbor_size - 1) // 2
    neighbor_offsets = []
    for offset_x in range(-neighbors_per_direction,
                          neighbors_per_direction + 1):
        for offset_y in range(-neighbors_per_direction,
                              neighbors_per_direction + 1):
            if offset_x == 0 and offset_y == 0:
                continue  # skip center
            neighbor_offsets.append([offset_x, offset_y])
    neighbor_offsets = np.asarray(neighbor_offsets, dtype=np.int32)

    len_x = grid_dims[0]
    len_y = grid_dims[1]
    filters_total = len_x * len_y

    # get neighbors for each filter
    neighbor_lists = []
    for ci in range(filters_total):
        neighbors = []
        # derive x and y coordinate in filter space
        cy = ci // len_x
        cx = ci % len_x
        for offset in neighbor_offsets:
            offset_x = cx + offset[0]
            offset_y = cy + offset[1]
            if 0 <= offset_x < len_x and 0 <= offset_y < len_y:
                # add neighbor if valid coordinate
                ni = offset_y * len_x + offset_x
                neighbors.append(ni)
        neighbor_lists.append(neighbors)

    # filter neighbor lists to only contain full neighborhoods
    center_ids = []
    neighbor_ids = []
    for ci, nis in enumerate(neighbor_lists):
        # e.g. in a 5x5 grid there are max. 24 neighbors
        if len(nis) == neighbor_size**2 - 1:
            center_ids.append(ci)
            neighbor_ids.append(nis)
    center_ids = np.asarray(center_ids, dtype=np.int32)
    neighbor_ids = np.asarray(neighbor_ids, dtype=np.int32)

    # weigh points further away in the neighborhood less
    neighbor_weights = []
    for offsets in neighbor_offsets:
        if weight_norm == "l1":
            d = np.abs(offsets).sum()
        elif weight_norm == "l2":
            d = np.sqrt((offsets*offsets).sum())
        elif weight_norm == "linf":
            d = np.abs(offsets).max()
        else:
            raise ValueError("Invalid weight norm specified: {}. "
                             "Valid are 'l1', 'l2', "
                             "'linf'.".format(weight_norm))
        w = 1. / d
        neighbor_weights.append(w)
    neighbor_weights = np.asarray(neighbor_weights, dtype=np.float32)
    neighbor_weights /= neighbor_weights.sum()  # normalize to sum=1
    # neighbor_weights /= np.sqrt((neighbor_weights ** 2).sum())  # normalize to length=1

    # now convert numpy arrays to tf constants
    with tf.name_scope("nd_regularizer_sebastian"):
        tf_neighbor_weights = tf.constant(neighbor_weights,
                                          name='neighbor_weights')
        tf_center_ids = tf.constant(center_ids, name='center_ids')
        tf_neighbor_ids = tf.constant(neighbor_ids, name='neighbor_ids')

        def neighbor_distance(filters):
            n_filters = filters.shape.as_list()[-1]
            if n_filters != filters_total:
                raise ValueError(
                    "Unsuitable grid for weight {}. "
                    "Grid dimensions: {}, {} for a total of {} entries. "
                    "Filters in weight: {}.".format(
                        filters.name, len_x, len_y, filters_total, n_filters))
            # filters to n_filters x d
            filters = tf.reshape(filters, [-1, n_filters])
            filters = tf.transpose(filters)

            # broadcast to n_centers x 1 x d
            tf_centers = tf.gather(filters, tf_center_ids)
            tf_centers = tf.expand_dims(tf_centers, 1)

            # n_centers x n_neighbors x d
            tf_neighbors = tf.gather(filters, tf_neighbor_ids)

            # compute pairwise distances, then weight, then sum up
            # pairwise is always n_centers x n_neighbors
            if diff_norm == "l1":
                pairwise = tf.reduce_sum(tf.abs(tf_centers - tf_neighbors),
                                         axis=-1)
            elif diff_norm == "l2":
                pairwise = tf.sqrt(
                    tf.reduce_sum((tf_centers - tf_neighbors)**2, axis=-1))
            elif diff_norm == "linf":
                pairwise = tf.reduce_max(tf.abs(tf_centers - tf_neighbors),
                                         axis=-1)
            elif diff_norm == "cos":
                dotprods = tf.reduce_sum(tf_centers * tf_neighbors, axis=-1)
                center_norms = tf.norm(tf_centers, axis=-1)
                neighbor_norms = tf.norm(tf_neighbors, axis=-1)
                # NOTE this computes cosine *similarity* which is why we
                # multiply by -1: minimize the negative similarity!
                cosine_similarity = dotprods / (center_norms * neighbor_norms)
                pairwise = -1 * cosine_similarity
            else:
                raise ValueError("Invalid difference norm specified: {}. "
                                 "Valid are 'l1', 'l2', 'linf', "
                                 "'cos'.".format(weight_norm))
            pairwise_weighted = tf_neighbor_weights * pairwise
            return tf.reduce_sum(pairwise_weighted) / \
                pairwise_weighted.shape.as_list()[0]

    return neighbor_distance


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
