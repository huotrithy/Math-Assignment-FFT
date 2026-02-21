import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, uniform_filter1d
import io

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Audio Denoising Lab", layout="wide")

st.title("🔊 FFT Audio Denoising")
# st.markdown(
#     "Compare three classical STFT-based speech enhancement algorithms."
# )

# --------------------------------------------------
# UTILITY
# --------------------------------------------------

def compute_snr(original, cleaned):
    noise = original - cleaned
    return 10 * np.log10(np.sum(cleaned**2) / (np.sum(noise**2) + 1e-10))


# --------------------------------------------------
# 1️⃣ Spectral Subtraction (Magnitude Domain)
# --------------------------------------------------

def spectral_subtraction_magnitude(y, sr, alpha, beta):
    n_fft = 2048
    hop_length = 512

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)

    # Estimate stationary noise (first 0.5 sec)
    noise_mu = np.mean(
        magnitude[:, :int(librosa.time_to_frames(0.5, sr=sr))],
        axis=1,
        keepdims=True,
    )

    mask = (magnitude - alpha * noise_mu) / (magnitude + 1e-10)
    mask = np.clip(mask, beta, 1.0)

    # Temporal smoothing
    mask = uniform_filter1d(mask, size=7, axis=1)

    cleaned_mag = magnitude * mask
    y_clean = librosa.istft(cleaned_mag * phase,
                        hop_length=hop_length,
                        length=len(y))

    return librosa.util.normalize(y_clean)


# --------------------------------------------------
# 2️⃣ Wiener Filter (Power Domain)
# --------------------------------------------------

def wiener_filter_power(y, sr, beta):
    n_fft = 2048
    hop_length = 512

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)

    power_spec = magnitude ** 2

    # Noise estimate (first 0.5 sec)
    noise_power = np.mean(
        power_spec[:, :int(librosa.time_to_frames(0.5, sr=sr))],
        axis=1,
        keepdims=True,
    )

    # Wiener gain
    mask = power_spec / (power_spec + noise_power + 1e-10)

    mask = np.clip(mask, beta, 1.0)
    mask = uniform_filter1d(mask, size=7, axis=1)

    cleaned_mag = magnitude * mask
    y_clean = librosa.istft(cleaned_mag * phase, hop_length=hop_length)

    return librosa.util.normalize(y_clean)


# --------------------------------------------------
# 3️⃣ Adaptive Power Spectral Subtraction
# --------------------------------------------------

def adaptive_power_subtraction(y, sr, alpha, beta):
    n_fft = 2048
    hop_length = 512

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)

    power_spec = magnitude ** 2

    # Adaptive median noise estimation
    window_size = int(
        librosa.time_to_frames(1.5, sr=sr, hop_length=hop_length)
    )

    noise_est = median_filter(power_spec, size=(1, window_size))

    subtracted = power_spec - alpha * noise_est

    mask = np.maximum(subtracted, beta * power_spec) / (power_spec + 1e-10)
    mask = np.sqrt(np.clip(mask, 0, 1))

    # Smooth in time and frequency
    mask = uniform_filter1d(mask, size=5, axis=1)
    mask = uniform_filter1d(mask, size=3, axis=0)

    cleaned_mag = magnitude * mask
    y_clean = librosa.istft(cleaned_mag * phase, hop_length=hop_length)

    return librosa.util.normalize(y_clean)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    method = st.selectbox(
        "Select Algorithm",
        [
            "Spectral Subtraction",
            "Wiener Filter",
            "Adaptive Power Spectral Subtraction",
        ],
    )

    # st.markdown("### Parameters")

    # alpha = st.slider("Over-subtraction (alpha)", 0.5, 5.0, 1.5, 0.1)
    # beta = st.slider("Spectral Floor (beta)", 0.0, 0.5, 0.1, 0.01)

    run_button = st.button("Apply Denoising")


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "ogg", "flac"],
)

if uploaded_file is not None:

    y_noisy, sr = librosa.load(uploaded_file, sr=None)

    st.subheader("Original Audio")
    st.audio(uploaded_file)

    if run_button:

        with st.spinner("Processing..."):

            if method == "Spectral Subtraction (Magnitude Domain)":
                y_clean = spectral_subtraction_magnitude(
                    y_noisy, sr, 1.5 , 0.15 
                )

            elif method == "Wiener Filter (Power Domain)":
                y_clean = wiener_filter_power(
                    y_noisy, sr, 0.15
                )

            else:
                y_clean = adaptive_power_subtraction(
                    y_noisy, sr, 3.0, 0.05
                )

        # --------------------------------------------------
        # CLEAN AUDIO OUTPUT
        # --------------------------------------------------

        buffer = io.BytesIO()
        sf.write(buffer, y_clean, sr, format="WAV")

        st.subheader("Cleaned Audio")
        st.audio(buffer.getvalue())

        # --------------------------------------------------
        # METRIC
        # --------------------------------------------------

        # snr_value = compute_snr(y_noisy, y_clean)
        # st.metric("Estimated Relative SNR", f"{snr_value:.2f} dB")

        # --------------------------------------------------
        # WAVEFORM
        # --------------------------------------------------

        # st.subheader("Waveform Comparison")

        # fig_wave, ax_wave = plt.subplots(2, 1, figsize=(12, 4), sharex=True)

        # librosa.display.waveshow(y_noisy, sr=sr, ax=ax_wave[0])
        # ax_wave[0].set_title("Original")

        # librosa.display.waveshow(y_clean, sr=sr, ax=ax_wave[1])
        # ax_wave[1].set_title("Cleaned")

        # st.pyplot(fig_wave)

        # --------------------------------------------------
        # SPECTROGRAMS
        # --------------------------------------------------

        st.subheader("Spectrogram Comparison")

        fig_spec, ax_spec = plt.subplots(1, 2, figsize=(18, 6))

        D_noisy = librosa.amplitude_to_db(
            np.abs(librosa.stft(y_noisy)), ref=np.max
        )

        D_clean = librosa.amplitude_to_db(
            np.abs(librosa.stft(y_clean)), ref=np.max
        )
        # D_diff = D_noisy - D_clean

        librosa.display.specshow(
            D_noisy, sr=sr, x_axis="time", y_axis="hz", ax=ax_spec[0]
        )
        ax_spec[0].set_title("Original")
        ax_spec[0].set_ylim(0, 4000)

        librosa.display.specshow(
            D_clean, sr=sr, x_axis="time", y_axis="hz", ax=ax_spec[1]
        )
        ax_spec[1].set_title("Cleaned")
        ax_spec[1].set_ylim(0, 4000)

        # librosa.display.specshow(
        #     D_diff, sr=sr, x_axis="time", y_axis="hz", ax=ax_spec[2]
        # )
        # ax_spec[2].set_title("Difference")

        st.pyplot(fig_spec)

        # --------------------------------------------------
        # INFO BOX
        # --------------------------------------------------

        # st.info(
        #     f"""
        #     **{method}** operates in the STFT domain.
            
        #     - Noise is estimated from signal statistics.
        #     - A frequency-dependent gain mask is applied.
        #     - Signal reconstructed via inverse STFT.
            
        #     These are classical DSP speech enhancement methods 
        #     (no deep learning involved).
        #     """
        # )
