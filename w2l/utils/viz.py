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
    librosa.display.specshow(phase, sr=sr, hop_length=hop_length, x_axis="time",
                             y_axis="linear", cmap="twilight_shifted")
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
