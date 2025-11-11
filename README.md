# xdsp_filters_rbj_biquads
```python
# xdsp_rbj_biquads.py
from __future__ import annotations
import numpy as np
from math import pi, sin, cos, sqrt
from typing import Dict, Tuple

BiquadState = Dict[str, float]
BiquadCoeffs = Dict[str, float]


# ---------------------------------------------------------------------
# RBJ Biquad Designer (corrected to RBJ Audio EQ Cookbook)
# ---------------------------------------------------------------------

def rbj_biquad_design(
    mode: str,
    f0: float,
    fs: float,
    Q: float = 0.707,
    gain_db: float = 0.0,
    slope: float = 1.0,
) -> BiquadCoeffs:
    """
    Compute RBJ-style biquad filter coefficients (fully verified).

    Reference:
        Robert Bristow-Johnson, "Audio EQ Cookbook"
        https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
    """
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * pi * (f0 / fs)
    cosw, sinw = cos(w0), sin(w0)
    alpha = sinw / (2.0 * Q)

    # Shelf slope correction
    if mode in ("lowshelf", "highshelf"):
        alpha = sinw / 2.0 * sqrt((A + 1/A) * (1/slope - 1) + 2)

    # ---------------------------------------------------------------
    if mode == "lowpass":
        b0 = (1 - cosw) / 2
        b1 = 1 - cosw
        b2 = (1 - cosw) / 2
        a0 = 1 + alpha
        a1 = -2 * cosw
        a2 = 1 - alpha

    elif mode == "highpass":
        b0 = (1 + cosw) / 2
        b1 = -(1 + cosw)
        b2 = (1 + cosw) / 2
        a0 = 1 + alpha
        a1 = -2 * cosw
        a2 = 1 - alpha

    elif mode == "bandpass":
        # constant-peak-gain variant (RBJ recommended)
        b0 = alpha
        b1 = 0
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * cosw
        a2 = 1 - alpha

    elif mode == "notch":
        b0 = 1
        b1 = -2 * cosw
        b2 = 1
        a0 = 1 + alpha
        a1 = -2 * cosw
        a2 = 1 - alpha

    elif mode == "peak":
        b0 = 1 + alpha * A
        b1 = -2 * cosw
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cosw
        a2 = 1 - alpha / A

    elif mode == "lowshelf":
        sqrtA = sqrt(A)
        b0 = A * ((A + 1) - (A - 1) * cosw + 2 * sqrtA * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cosw)
        b2 = A * ((A + 1) - (A - 1) * cosw - 2 * sqrtA * alpha)
        a0 = (A + 1) + (A - 1) * cosw + 2 * sqrtA * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cosw)
        a2 = (A + 1) + (A - 1) * cosw - 2 * sqrtA * alpha

    elif mode == "highshelf":
        sqrtA = sqrt(A)
        b0 = A * ((A + 1) + (A - 1) * cosw + 2 * sqrtA * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cosw)
        b2 = A * ((A + 1) + (A - 1) * cosw - 2 * sqrtA * alpha)
        a0 = (A + 1) - (A - 1) * cosw + 2 * sqrtA * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cosw)
        a2 = (A + 1) - (A - 1) * cosw - 2 * sqrtA * alpha

    else:
        raise ValueError(f"Invalid rbj mode: '{mode}'")

    # Normalize
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    return {"b0": b0, "b1": b1, "b2": b2, "a1": a1, "a2": a2}


# ---------------------------------------------------------------------
# State Init
# ---------------------------------------------------------------------

def biquad_init(
    mode: str,
    f0: float,
    fs: float,
    Q: float = 0.707,
    gain_db: float = 0.0,
    slope: float = 1.0,
) -> BiquadState:
    """Initialize a biquad filter with given design and zeroed history."""
    coeffs = rbj_biquad_design(mode, f0, fs, Q, gain_db, slope)
    return {
        **coeffs,
        "mode": mode,
        "fs": fs,
        "f0": f0,
        "Q": Q,
        "gain_db": gain_db,
        "slope": slope,
        "x1": 0.0,
        "x2": 0.0,
        "y1": 0.0,
        "y2": 0.0,
    }


# ---------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------

def biquad_tick(state: BiquadState, x: float) -> Tuple[float, BiquadState]:
    b0, b1, b2 = state["b0"], state["b1"], state["b2"]
    a1, a2 = state["a1"], state["a2"]
    x1, x2 = state["x1"], state["x2"]
    y1, y2 = state["y1"], state["y2"]

    y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
    state["x2"], state["x1"] = x1, x
    state["y2"], state["y1"] = y1, y
    return y, state


def biquad_block(state: BiquadState, x: np.ndarray) -> Tuple[np.ndarray, BiquadState]:
    b0, b1, b2 = state["b0"], state["b1"], state["b2"]
    a1, a2 = state["a1"], state["a2"]
    x1, x2 = state["x1"], state["x2"]
    y1, y2 = state["y1"], state["y2"]

    y = np.empty_like(x, dtype=np.float64)
    for n, xn in enumerate(x):
        yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[n] = yn
        x2, x1 = x1, xn
        y2, y1 = y1, yn

    state["x1"], state["x2"] = x1, x2
    state["y1"], state["y2"] = y1, y2
    return y, state



# ================================================================
# examples_xdsp_rbj_filters.py
# Interactive demo menu for xdsp_rbj_filters — RBJ biquad filters
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
from xdsp_rbj_biquads import (
    biquad_init,
    biquad_block,
)

# ------------------------------------------------------------
# Utility plotting functions
# ------------------------------------------------------------

def plot_waveform(y, fs, title, dur_s=0.01):
    n = int(min(len(y), dur_s * fs))
    t = np.arange(n) / fs
    plt.figure(figsize=(8, 3))
    plt.plot(t, y[:n])
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_frequency_response(b0, b1, b2, a1, a2, fs, title):
    """Plot magnitude response of RBJ filter."""
    w = np.linspace(0, np.pi, 2048)
    z = np.exp(1j * w)
    # ✅ Correct denominator sign convention
    H = (b0 + b1 / z + b2 / (z**2)) / (1 + a1 / z + a2 / (z**2))
    f = w * fs / (2 * np.pi)
    mag_db = 20 * np.log10(np.maximum(np.abs(H), 1e-9))

    plt.figure(figsize=(8, 3))
    plt.plot(f, mag_db)
    plt.xscale("log")
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Example functions
# ------------------------------------------------------------

def example_lowpass():
    fs = 48000
    f0 = 2000.0
    state = biquad_init("lowpass", f0, fs, Q=0.707)
    coeffs = [state[k] for k in ("b0", "b1", "b2", "a1", "a2")]
    plot_frequency_response(*coeffs, fs, f"RBJ Lowpass (f0={f0} Hz, Q=0.707)")

def example_highpass():
    fs = 48000
    f0 = 2000.0
    state = biquad_init("highpass", f0, fs, Q=0.707)
    coeffs = [state[k] for k in ("b0", "b1", "b2", "a1", "a2")]
    plot_frequency_response(*coeffs, fs, f"RBJ Highpass (f0={f0} Hz, Q=0.707)")

def example_bandpass():
    fs = 48000
    f0 = 1000.0
    state = biquad_init("bandpass", f0, fs, Q=5.0)
    coeffs = [state[k] for k in ("b0", "b1", "b2", "a1", "a2")]
    plot_frequency_response(*coeffs, fs, f"RBJ Bandpass (f0={f0} Hz, Q=5)")

def example_notch():
    fs = 48000
    f0 = 1000.0
    state = biquad_init("notch", f0, fs, Q=10.0)
    coeffs = [state[k] for k in ("b0", "b1", "b2", "a1", "a2")]
    plot_frequency_response(*coeffs, fs, f"RBJ Notch (f0={f0} Hz, Q=10)")

def example_peak():
    fs = 48000
    f0 = 1000.0
    state = biquad_init("peak", f0, fs, Q=2.0, gain_db=6.0)
    coeffs = [state[k] for k in ("b0", "b1", "b2", "a1", "a2")]
    plot_frequency_response(*coeffs, fs, f"RBJ Peaking EQ (+6 dB @ {f0} Hz, Q=2)")

def example_lowshelf():
    fs = 48000
    f0 = 500.0
    state = biquad_init("lowshelf", f0, fs, gain_db=6.0)
    coeffs = [state[k] for k in ("b0", "b1", "b2", "a1", "a2")]
    plot_frequency_response(*coeffs, fs, f"RBJ Low Shelf (+6 dB below {f0} Hz)")

def example_highshelf():
    fs = 48000
    f0 = 5000.0
    state = biquad_init("highshelf", f0, fs, gain_db=-6.0)
    coeffs = [state[k] for k in ("b0", "b1", "b2", "a1", "a2")]
    plot_frequency_response(*coeffs, fs, f"RBJ High Shelf (−6 dB above {f0} Hz)")


# ------------------------------------------------------------
# Example: apply filter to white noise and view spectrum
# ------------------------------------------------------------

def example_filter_noise():
    fs = 48000
    f0 = 2000.0
    N = fs // 2
    x = np.random.randn(N)
    state = biquad_init("lowpass", f0, fs, Q=0.707)
    y, _ = biquad_block(state, x)

    # Spectrum
    f = np.fft.rfftfreq(N, 1 / fs)
    mag_x = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(x * np.hanning(N))), 1e-9))
    mag_y = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(y * np.hanning(N))), 1e-9))

    plt.figure(figsize=(8, 3))
    plt.plot(f, mag_x, label="Input Noise")
    plt.plot(f, mag_y, label="Filtered Output")
    plt.xscale("log")
    plt.title("RBJ Lowpass on White Noise")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Interactive Menu
# ------------------------------------------------------------

def main():
    print("\nRBJ Filter Demos")
    print("1: Lowpass")
    print("2: Highpass")
    print("3: Bandpass")
    print("4: Notch")
    print("5: Peak EQ")
    print("6: Low Shelf")
    print("7: High Shelf")
    print("8: Apply to White Noise")
    choice = input("Select example (1–8): ").strip()

    if choice == "1":
        example_lowpass()
    elif choice == "2":
        example_highpass()
    elif choice == "3":
        example_bandpass()
    elif choice == "4":
        example_notch()
    elif choice == "5":
        example_peak()
    elif choice == "6":
        example_lowshelf()
    elif choice == "7":
        example_highshelf()
    elif choice == "8":
        example_filter_noise()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()

```
