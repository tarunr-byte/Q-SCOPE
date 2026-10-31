"""
physics/spectrogram_utils.py
============================
Spectrogram generation utilities for Azhva Diagnostic AI Agent.
Converts 1D analog waveforms into Base64-encoded images for visual AI models.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import List

import numpy as np
import matplotlib

# Force non-interactive backend to prevent crashes in FastAPI threads
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

logger = logging.getLogger(__name__)


def generate_spectrogram_base64(samples: List[float], sampling_rate_hz: float = 200.0) -> str:
    """
    Generates a high-contrast spectrogram image from a 1D signal and returns it as a Base64 string.

    Args:
        samples: The 1D cleaned signal array.
        sampling_rate_hz: The ADC sampling rate in Hz.

    Returns:
        A Base64 encoded PNG string. Returns empty string on failure.
    """
    logger.info("[Spectrogram] Generating image for signal of length %d", len(samples))

    if not samples or len(samples) < 16:
        logger.warning("Signal too short for spectrogram analysis.")
        return ""

    fig = None
    try:
        # Compute the Spectrogram
        nperseg = min(len(samples), 64)
        noverlap = min(nperseg // 2, 32)
        f, t, Sxx = spectrogram(
            np.asarray(samples, dtype=float),
            fs=sampling_rate_hz,
            nperseg=nperseg,
            noverlap=noverlap,
        )

        # Plotting
        fig, ax = plt.subplots(figsize=(6, 3), dpi=100)

        # Colormap: inferno gives high contrast for frequency intensity
        ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='inferno')

        # Styling
        ax.set_title('Azhva HGV Telemetry Spectrogram', fontsize=10, pad=8, color='white')
        ax.set_ylabel('Frequency [Hz]', fontsize=8, color='white')
        ax.set_xlabel('Time [sec]', fontsize=8, color='white')
        ax.tick_params(axis='both', which='major', labelsize=7, colors='white')
        ax.set_facecolor('#0f172a')
        fig.patch.set_facecolor('#0a0f1d')

        # Save to memory buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#0a0f1d')
        plt.close(fig)
        fig = None
        buf.seek(0)

        # Encode to Base64
        b64_string = base64.b64encode(buf.read()).decode('utf-8')
        logger.info("[Spectrogram] Image generated and Base64 encoded successfully.")
        return b64_string

    except Exception as exc:
        logger.error("[Spectrogram] Failed to generate image: %s", exc)
        if fig is not None:
            plt.close(fig)
        return ""


def extract_dominant_frequency(samples: List[float], sampling_rate_hz: float = 200.0) -> float:
    """Extract dominant peak frequency from raw analog signal."""
    try:
        if not samples or len(samples) < 16:
            return 0.0
        nperseg = min(len(samples), 64)
        f, _, Sxx = spectrogram(np.asarray(samples, dtype=float), fs=sampling_rate_hz, nperseg=nperseg)
        peak_freq = float(f[np.argmax(np.sum(Sxx, axis=1))])
        logger.info("[Spectrogram] Extracted dominant frequency: %.2f Hz", peak_freq)
        return peak_freq
    except Exception as exc:
        logger.error("Failed to extract dominant frequency: %s", exc)
        return 0.0
