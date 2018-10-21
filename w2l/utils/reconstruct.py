import librosa
import numpy as np


def reconstruct_from_mag_phase(mag, phase, hop_length=160):
    """Reconstruct audio from magnitude and phase.

    Parameters:
        mag: 2D linear (!) magnitude spectrogram, bins x time.
        phase: 2D phase spectrogram bins x time.
        hop_length: Hop length used to create the spectrograms.

    Returns:
        The reconstructed time domain signal as a 1-dim Numpy array.
    """
    rec = mag * np.exp(1.j * phase)
    rec = librosa.istft(rec, hop_length=hop_length, center=True)
    return rec


def griffin_lim(mag, iterations, window_size=400, hop_length=160,
                verbose=False):
    """Reconstruct an audio signal from a magnitude spectrogram.

    Given a magnitude spectrogram as input, reconstruct
    the audio signal and return it using the Griffin-Lim algorithm from the
    paper: "Signal estimation from modified short-time fourier transform" by
    Griffin and Lim, in IEEE transactions on Acoustics, Speech, and Signal
    Processing. Vol ASSP-32, No. 2, April 1984.

    Parameters:
        mag: 2D linear (!) magnitude spectrogram, bins x time.
        iterations: Number of iterations for the Griffin-Lim algorithm.
                    Typically a few hundred is sufficient.
        window_size: Window size used to create the spectrogram.
        hop_length: Hop length used to create the spectrogram.
        verbose: Set to print reconstruction error after each iteration.

    Returns:
        The reconstructed time domain signal as a 1-dim Numpy array.
    """
    time_slices = mag.shape[1]
    len_samples = int((time_slices-1)*hop_length)
    # Initialize the reconstructed signal to noise
    x_reconstruct = np.random.randn(len_samples)
    for ii in range(iterations):
        recon_spectro = librosa.stft(x_reconstruct, n_fft=window_size,
                                     hop_length=hop_length)
        recon_angle = np.angle(recon_spectro)
        # Discard magnitude part of the reconstruction and use the supplied
        # magnitude spectrogram instead
        proposal_spectrogram = mag * np.exp(1.0j * recon_angle)
        prev_x = x_reconstruct
        x_reconstruct = librosa.istft(proposal_spectrogram,
                                      hop_length=hop_length)
        diff = np.sqrt(np.sum((x_reconstruct - prev_x)**2)/x_reconstruct.size)
        if verbose or not (ii + 1) % 100:
            print('Reconstruction iteration: {}/{} RMSE: {} '.format(
                ii+1, iterations, diff))
    return x_reconstruct


def mel_to_linear(mel, log=False, power=False, sr=16000, window_size=400):
    """Approximately invert a mel spectrogram to linear frequency bins.

    Parameters:
        mel: The mel spectrogram.
        log: If true, interpret input as log mel spectrogram.
        power: If true, interpret mag as a power spectrogram.
        sr: Sampling rate of the audio data.
        window_size: Window size used to create the spectrogram.

    Returns:
        Linear frequency spectrogram.
    """
    if log:
        mel = np.exp(mel)

    mel_basis = librosa.filters.mel(sr=sr, n_fft=window_size,
                                    n_mels=mel.shape[0])
    linear = np.dot(mel_basis.T, mel)

    if power:
        linear = np.sqrt(linear)
    return linear
