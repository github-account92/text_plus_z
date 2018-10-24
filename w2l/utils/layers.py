import numpy as np
import tensorflow as tf


def cnn1d_from_config(parsed_config, inputs, act, batchnorm, train, data_format,
                      vis, transpose=False, prefix=""):
    """Build 1D convolutional models without pooling like Wav2letter.

    The final layer should *not* be included since it's always the same and
    depends on the data (i.e. vocabulary size).

    Parameters:
        parsed_config: Result of parsing a model config file, containing only
                       the relevant CNN stuff.
        inputs: 3D inputs to the model (audio).
        act: Activation function to apply in each layer/block.
        batchnorm: Bool, whether to use batch normalization.
        train: Bool or TF placeholder. Fed straight into batch normalization
               (ignored if that is not used).
        data_format: channels_first or _last. Assumed that you checked validity
                     beforehand. I.e. if it's not first, this function simply
                     assumes that it's last.
        vis: Bool, whether to add histograms for layer activations.
        transpose: If true, use transposed convolutions.
        prefix: String to prepend to all layer names (also creates variable
                scope).
    """
    previous = inputs
    total_pars = 0
    all_layers = []
    total_stride = 1

    with tf.variable_scope(prefix):
        for ind, (_type, n_f, w_f, s_f, d_f) in enumerate(parsed_config):
            name = "encoder_" + _type + str(ind)
            if _type == "layer":
                if transpose:
                    previous, pars = transposed_conv_layer(
                        previous, n_f, w_f, s_f, d_f, act, batchnorm, train,
                        data_format, vis, name=name)
                else:
                    previous, pars = conv_layer(
                        previous, n_f, w_f, s_f, d_f, act, batchnorm, train,
                        data_format, vis, name=name)
            # TODO residual/dense blocks ignore some parameters ATM!
            elif _type == "block":
                previous, pars = residual_block(
                    previous, n_f, w_f, s_f, act, batchnorm, train,
                    data_format, vis, name=name)
            elif _type == "dense":
                previous, pars = dense_block(
                    previous, n_f, w_f, s_f, act, batchnorm, train,
                    data_format, vis, name=name)
            else:
                raise ValueError(
                    "Invalid layer type specified in layer {}! Valid are "
                    "'layer', 'block', 'dense'. You specified "
                    "{}.".format(ind, _type))
            all_layers.append((name, previous))
            total_stride *= s_f
            total_pars += pars

    return previous, total_stride, all_layers, total_pars


def conv_layer(inputs, n_filters, width_filters, stride_filters, dilation, act,
               batchnorm, train, data_format, vis, name):
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
        name: Name of the layer (used for variable scope and summary).

    Returns:
        Output of the layer and number of parameters.
    """
    channel_axis = 1 if data_format == "channels_first" else -1

    with tf.variable_scope(name):
        conv = tf.layers.Conv1D(
            filters=n_filters, kernel_size=width_filters,
            strides=stride_filters, dilation_rate=dilation,
            activation=None if batchnorm else act,
            use_bias=not batchnorm, padding="same", data_format=data_format,
            name="conv")
        conved = conv.apply(inputs)
        n_pars = sum([np.prod(weight.shape.as_list()) for
                      weight in conv.trainable_weights])

        if batchnorm:
            bn = tf.layers.BatchNormalization(
                axis=channel_axis, name="batch_norm")
            conved = bn.apply(conved, training=train)
            n_pars += sum([np.prod(weight.shape.as_list()) for
                           weight in bn.trainable_weights])
            if act:
                conved = act(conved)
        if vis:
            tf.summary.histogram("activations_" + name, conved)

        print("\tCreated layer {} with {} parameters...".format(name, n_pars))
        print("\t\tChannels: {} Filters: {} Width: {} Stride: {}, "
              "Dilation: {}".format(
                  inputs.shape.as_list()[channel_axis], n_filters,
                  width_filters, stride_filters, dilation))
        return conved, n_pars


def transposed_conv_layer(inputs, n_filters, width_filters, stride_filters,
                          dilation, act, batchnorm, train, data_format, vis,
                          name):
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
        name: Name of the layer (used for variable scope and summary).

    Returns:
        Output of the layer and number of parameters.
    """
    channel_axis = 1 if data_format == "channels_first" else -1

    with tf.variable_scope(name):
        # for some reason there is no 1d transposed conv layer
        # b x c x t -> b x c x t x 1
        # b x t x c -> b x t x 1 x c
        inp_2d = tf.expand_dims(
            inputs, axis=3 if data_format == "channels_first" else 2)
        conv = tf.layers.Conv2DTranspose(
            filters=n_filters, kernel_size=(width_filters, 1),
            strides=(stride_filters, 1), activation=None if batchnorm else act,
            use_bias=not batchnorm, padding="same", data_format=data_format,
            name="conv")
        conved = conv.apply(inp_2d)
        if data_format == "channels_first":
            conved = tf.squeeze(conved, axis=3)
        else:
            conved = tf.squeeze(conved, axis=2)
        n_pars = sum([np.prod(weight.shape.as_list()) for
                      weight in conv.trainable_weights])

        if batchnorm:
            bn = tf.layers.BatchNormalization(
                axis=channel_axis, name="batch_norm")
            conved = bn.apply(conved, training=train)
            n_pars += sum([np.prod(weight.shape.as_list()) for
                           weight in bn.trainable_weights])
            if act:
                conved = act(conved)
        if vis:
            tf.summary.histogram("activations_" + name, conved)

        print("\tCreated layer {} with {} parameters...".format(name, n_pars))
        print("\t\tChannels: {} Filters: {} Width: {} Stride: {}, "
              "Dilation: {}".format(
                   inputs.shape.as_list()[channel_axis], n_filters,
                   width_filters, stride_filters, dilation))
        return conved, n_pars


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
