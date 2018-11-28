"""Visualization wrappers."""
import string

import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt


def fig_mel_spec(spectro, sr=16000, hop_length=160, labelsize=18, ticksize=15,
                 show=True, store=""):
    """Plot a mel spectrogram with reasonable default parameters.

    Parameters:
        spectro: 2D spectrogram, frequency x time.
        sr: Sampling rate of the data.
        hop_length: Hop length of the data.
        labelsize: Font size to use for axis labels.
        ticksize: Font size to use for tick labels.
        show: If True, display the plot.
        store: Path where the figure should be stored. If not given, nothing is
               stored.

    """
    librosa.display.specshow(spectro, sr=sr, hop_length=hop_length,
                             x_axis="time", y_axis="mel", fmax=sr//2)
    plt.tick_params(axis="y", labelsize=ticksize)
    plt.tick_params(axis="x", labelsize=ticksize)
    plt.xlabel("Time (seconds)", fontsize=labelsize)
    plt.ylabel("Frequency (Hz)", fontsize=labelsize)
    if store:
        plt.savefig(store, bbox_inches="tight")
    if show:
        plt.show()


def fig_phase(phase, sr=16000, hop_length=160, labelsize=18, ticksize=15,
              show=True, store=""):
    """Plot a phase spectrogram with reasonable default parameters.

    Parameters:
        phase: 2D phase angle spectrogram, frequency x time.
        sr: Sampling rate of the data.
        hop_length: Hop length of the data.
        labelsize: Font size to use for axis labels.
        ticksize: Font size to use for tick labels.
        show: If True, display the plot.
        store: Path where the figure should be stored. If not given, nothing is
               stored.

    """
    librosa.display.specshow(phase, sr=sr, hop_length=hop_length,
                             x_axis="time", y_axis="linear",
                             cmap="twilight_shifted")
    plt.tick_params(axis="y", labelsize=ticksize)
    plt.tick_params(axis="x", labelsize=ticksize)
    plt.xlabel("Time (seconds)", fontsize=labelsize)
    plt.ylabel("Frequency (Hz)", fontsize=labelsize)
    if store:
        plt.savefig(store, bbox_inches="tight")
    if show:
        plt.show()


def fig_wave(wave, sr=16000, labelsize=18, ticksize=15, show=True, store=""):
    """Plot a wave signal with reasonable default parameters.

    Parameters:
        wave: The thing to plot.
        sr: Sampling rate of the data.
        labelsize: Font size to use for axis labels.
        ticksize: Font size to use for tick labels.
        show: If True, display the plot.
        store: Path where the figure should be stored. If not given, nothing is
               stored.

    """
    librosa.display.waveplot(wave, sr=sr)
    plt.tick_params(axis="y", labelsize=ticksize)
    plt.tick_params(axis="x", labelsize=ticksize)
    plt.xlabel("Time (seconds)", fontsize=labelsize)
    plt.ylabel("Amplitude", fontsize=labelsize)
    if store:
        plt.savefig(store, bbox_inches='tight')
    if show:
        plt.show()


def fig_logits(logits, sr=16000, hop_length=320, labelsize=18, ticksize=15,
               show=True, store=""):
    """Plot the logit space with reasonable default parameters.

    Parameters:
        logits: The logits to display.
        sr: Sampling rate of the data.
        hop_length: Hop length of the data. Note that this needs to include
                    any striding in the network.
        labelsize: Font size to use for axis labels.
        ticksize: Font size to use for tick labels.
        show: If True, display the plot.
        store: Path where the figure should be stored. If not given, nothing is
               stored.

    """
    librosa.display.specshow(logits, sr=sr, hop_length=hop_length,
                             x_axis="time", cmap="magma")
    char_ticks = list(range(29))
    char_ticks = [t + 0.5 for t in char_ticks]
    char_tick_labels = ["SP", "AP"] + list(string.ascii_lowercase) + ["BL"]
    plt.yticks(char_ticks[1::2], char_tick_labels[1::2])
    plt.tick_params(axis="y", length=1, labelsize=ticksize, width=0)
    plt.xticks([])
    plt.xlabel("Time (seconds)", fontsize=labelsize)

    ya = plt.twinx()
    ya.set_yticks([d for d in char_ticks[::2]], minor=False)
    ya.set_yticklabels(char_tick_labels[::2], minor=False)
    plt.tick_params(axis="y", which="major", length=1, labelsize=ticksize,
                    width=0)
    if store:
        plt.savefig(store, bbox_inches="tight")
    if show:
        plt.show()


def fig_latent(latent, sr=16000, hop_length=320, labelsize=18, ticksize=15,
               show=True, store="", diverging=True):
    """Plot the logit space with reasonable default parameters.

    Parameters:
        latent: The logits to display.
        sr: Sampling rate of the data.
        hop_length: Hop length of the data. Note that this needs to include
                    any striding in the network.
        labelsize: Font size to use for axis labels.
        ticksize: Font size to use for tick labels.
        show: If True, display the plot.
        store: Path where the figure should be stored. If not given, nothing is
               stored.
        diverging: If true, use a diverging colormap (coolwarm). Otherwise use
                   a sequential one (magma)

    """
    if diverging:
        absmax = np.max(np.abs(latent))
        kwargs = {"cmap": "RdBu_r", "vmin": -absmax, "vmax": absmax}
    else:
        kwargs = {"cmap": "magma"}
    librosa.display.specshow(latent, sr=sr, hop_length=hop_length,
                             x_axis="time", **kwargs)
    plt.tick_params(axis="y", labelsize=ticksize)
    plt.tick_params(axis="x", labelsize=ticksize)
    plt.xlabel("Time (seconds)", fontsize=labelsize)
    plt.ylabel("Latent dimension", fontsize=labelsize)
    if store:
        plt.savefig(store, bbox_inches="tight")
    if show:
        plt.show()


def img_grid_npy(imgs, dims, borders=(1, 1), border_val=None, normalize=True):
    """
    Pack a bunch of image-like arrays into a grid for nicer visualization.

    Parameters:
        imgs: Should be a list of arrays, each height x width. E.g. if you have
              32 filter maps, i.e. array height x width x 32 you should pack
              those into a list of 32 elements of size height x width.
              Alternatively you can just have a 32 x height x with array.
        dims: Tuple, how many rows/columns to use for the grid. Product has to
              match the number of images!
        borders: Size of the borders between images. Tuples; first entry gives
                 row borders and second entry column borders.
        border_val: What value to use for the borders inbetween images. If not
                    given, the maximum over all images will be used.
        normalize: If True, normalize each image separately to absolute maximum
                   of 1. Only scaled, not shifted!

    Returns:
         A 2d tensor you can use for matplotlib.
    """
    rows, cols = dims
    if len(imgs) != rows * cols:
        raise ValueError("Grid doesn't match the number of images!")

    if normalize:
        def norm(img):
            return img / np.abs(img).max()

        imgs = np.array([norm(img) for img in imgs])

    if border_val is None:
        border_val = imgs.max()

    # make border things
    col_border = np.full([imgs[0].shape[0], borders[1]], border_val)

    # first create the rows
    def make_row(ind):
        base = imgs[ind:(ind + cols)]
        _borders = [col_border] * len(base)
        _interleaved = [elem for pair in zip(base, _borders) for
                        elem in pair][:-1]  # remove last border
        return _interleaved

    grid_rows = [np.concatenate(make_row(ind), axis=1) for
                 ind in range(0, len(imgs), cols)]

    # then stack them
    row_border = np.full([borders[0], grid_rows[0].shape[1]], border_val)
    borders = [row_border] * len(grid_rows)
    interleaved = [elem for pair in zip(grid_rows, borders) for
                   elem in pair][:-1]
    grid = np.concatenate(interleaved)
    return grid
