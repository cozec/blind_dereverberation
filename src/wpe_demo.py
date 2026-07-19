"""Blind dereverberation demo using WPE (nara_wpe).

Pipeline:
1. Concatenate clean CMU Arctic utterances (16 kHz mono).
2. Synthesize a 4-mic reverberant recording by convolving the clean signal
   with synthetic room impulse responses (exponentially decaying noise,
   RT60 ~ 0.6 s, direct path at t=0 so signals stay time-aligned).
3. Run offline multi-channel WPE to blindly dereverberate.
4. Save audio (clean / reverberant / dereverberated), plot spectrograms,
   and report STOI against the clean reference.
"""

import os
import urllib.request

import numpy as np
import soundfile as sf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pystoi import stoi as stoi_fn

from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft

RNG = np.random.default_rng(0)
FS = 16000
RT60 = 0.6          # seconds
N_MICS = 4
STFT_SIZE, STFT_SHIFT = 512, 128
TAPS, DELAY, ITERATIONS = 10, 3, 5


def make_rir(rt60, fs, rng):
    """Synthetic RIR: unit direct impulse + exponentially decaying noise tail."""
    n = int(rt60 * fs)
    t = np.arange(n) / fs
    decay = np.exp(-3.0 * np.log(10) * t / rt60)  # -60 dB at rt60
    tail = rng.standard_normal(n) * decay
    tail[0] = 0.0
    rir = 0.25 * tail
    rir[0] = 1.0  # direct path at t=0 keeps clean/reverb aligned
    return rir


def spectrogram_db(x):
    X = stft(x[None, :], size=STFT_SIZE, shift=STFT_SHIFT)[0]
    return 20 * np.log10(np.abs(X).T + 1e-10)


ARCTIC_URL = "http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_bdl_arctic/wav"
UTTERANCES = ["arctic_a0001", "arctic_a0002", "arctic_a0003", "arctic_a0004"]


def fetch_data():
    """Download the CMU Arctic utterances into data/ if not already present."""
    os.makedirs("data", exist_ok=True)
    for name in UTTERANCES:
        path = f"data/{name}.wav"
        if not os.path.exists(path):
            print(f"Downloading {name}.wav ...")
            urllib.request.urlretrieve(f"{ARCTIC_URL}/{name}.wav", path)


def main():
    fetch_data()
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    clean = np.concatenate([sf.read(f"data/{f}.wav")[0] for f in UTTERANCES])
    clean = clean / np.max(np.abs(clean))
    n = len(clean)
    print(f"Clean signal: {n / FS:.1f} s at {FS} Hz")

    # Reverberant multi-channel observation
    obs = np.stack(
        [np.convolve(clean, make_rir(RT60, FS, RNG))[:n] for _ in range(N_MICS)]
    )
    obs = obs / np.max(np.abs(obs))

    # WPE: (channels, samples) -> STFT (F, D, T) -> WPE -> iSTFT
    Y = stft(obs, size=STFT_SIZE, shift=STFT_SHIFT).transpose(2, 0, 1)
    Z = wpe(Y, taps=TAPS, delay=DELAY, iterations=ITERATIONS).transpose(1, 2, 0)
    dereverb = istft(Z, size=STFT_SIZE, shift=STFT_SHIFT)[:, :n]

    sf.write("results/clean.wav", clean, FS)
    sf.write("results/reverberant.wav", obs[0], FS)
    out = dereverb[0] / np.max(np.abs(dereverb[0]))
    sf.write("results/dereverberated.wav", out, FS)

    stoi_rev = stoi_fn(clean, obs[0], FS)
    stoi_der = stoi_fn(clean, out, FS)
    print(f"STOI reverberant (mic 0):   {stoi_rev:.3f}")
    print(f"STOI dereverberated (WPE):  {stoi_der:.3f}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    titles = [
        "Clean",
        f"Reverberant (RT60 = {RT60} s), STOI = {stoi_rev:.3f}",
        f"WPE dereverberated, STOI = {stoi_der:.3f}",
    ]
    for ax, sig, title in zip(axes, [clean, obs[0], out], titles):
        S = spectrogram_db(sig)
        ax.imshow(
            S,
            origin="lower",
            aspect="auto",
            vmin=S.max() - 80,
            vmax=S.max(),
            cmap="magma",
            extent=[0, n / FS, 0, FS / 2000],
        )
        ax.set_title(title)
        ax.set_ylabel("kHz")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig("plots/spectrograms.png", dpi=120)
    print("Wrote results/*.wav and plots/spectrograms.png")


if __name__ == "__main__":
    main()
