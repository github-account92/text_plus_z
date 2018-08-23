import librosa
import numpy as np


def reconstruct_from_mag_phase(mag, phase, power=False, log=False, mel=False):
    """Reconstruct audio from magnitude and phase.

    Parameters:
        mag: 2D magnitude spectrogram, bins x time.
        phase: 2D phase spectrogram bins x time.
        power: If true, interpret mag as a power spectrogram.
        log: If true, interpret mag as a log spectrogram.
        mel: If true, interpret mag as a mel spectrogram.

    Returns:
        The reconstructed time domain signal as a 1-dim Numpy array.
    """
    if mel:
        mag = mel_to_linear(mag, log=log)
    elif log:  # if mel, this was done in mel_to_linear already
        mag = np.exp(mag)

    if power:
        mag = np.sqrt(mag)

    rec = mag * np.exp(1.j * phase)
    rec = librosa.istft(rec, hop_length=160, center=True)
    return rec


def griffin_lim(mag, iterations, verbose=False):
    """Reconstruct an audio signal from a magnitude spectrogram.

    Given a magnitude spectrogram as input, reconstruct
    the audio signal and return it using the Griffin-Lim algorithm from the
    paper: "Signal estimation from modified short-time fourier transform" by
    Griffin and Lim, in IEEE transactions on Acoustics, Speech, and Signal
    Processing. Vol ASSP-32, No. 2, April 1984.

    Parameters:
        mag: 2D magnitude spectrogram, bins x time.
        iterations: Number of iterations for the Griffin-Lim algorithm.
                    Typically a few hundred is sufficient.
        verbose: Set to print reconstruction error after each iteration.

    Returns:
        The reconstructed time domain signal as a 1-dim Numpy array.
    """
    time_slices = mag.shape[1]
    len_samples = int((time_slices-1)*160)
    # Initialize the reconstructed signal to noise
    x_reconstruct = np.random.randn(len_samples)
    print(x_reconstruct.shape, "OOH")
    for ii in range(iterations):
        recon_spectro = librosa.stft(x_reconstruct, n_fft=400, hop_length=160)
        recon_angle = np.angle(recon_spectro)
        # Discard magnitude part of the reconstruction and use the supplied
        # magnitude spectrogram instead
        proposal_spectrogram = mag * np.exp(1.0j * recon_angle)
        prev_x = x_reconstruct
        x_reconstruct = librosa.istft(proposal_spectrogram, hop_length=160)
        diff = np.sqrt(np.sum((x_reconstruct - prev_x)**2)/x_reconstruct.size)
        if verbose:
            print('Reconstruction iteration: {}/{} RMSE: {} '.format(
                ii+1, iterations, diff))
    return x_reconstruct


def mel_to_linear(mel, log=False):
    """Approximately invert a mel spectrogram to linear frequency bins.

    Parameters:
        mel: The mel spectrogram.
        log: If true, interpret input as log mel spectrogram.

    Returns:
        Linear frequency spectrogram.
    """
    if log:
        mel = np.exp(mel)

    mel_basis = librosa.filters.mel(sr=16000, n_fft=400, n_mels=mel.shape[0])
    return np.dot(mel_basis.T, mel)
