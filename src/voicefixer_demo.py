"""Neural restoration of the reverberant demo clip using voicefixer.

Run src/wpe_demo.py first (it creates results/clean.wav and
results/reverberant.wav). This script restores the reverberant mic signal
with voicefixer (mode 0), resamples the 44.1 kHz output back to 16 kHz,
aligns it to the clean reference by cross-correlation, and reports STOI
for comparison with the WPE result.
"""

import numpy as np
import soundfile as sf
import librosa
from pystoi import stoi as stoi_fn
from voicefixer import VoiceFixer

FS = 16000


def align(ref, x, max_lag=FS):
    """Shift x to best match ref using cross-correlation (within +/- 1 s)."""
    n = min(len(ref), len(x))
    corr = np.correlate(x[:n], ref[:n], mode="full")
    center = n - 1
    lag = np.argmax(corr[center - max_lag : center + max_lag]) - max_lag
    x = x[lag:] if lag > 0 else np.pad(x, (-lag, 0))
    n = min(len(ref), len(x))
    return ref[:n], x[:n]


def main():
    vf = VoiceFixer()
    vf.restore(
        input="results/reverberant.wav",
        output="results/voicefixer_44k.wav",
        cuda=False,
        mode=0,
    )

    out44, sr = sf.read("results/voicefixer_44k.wav")
    if out44.ndim > 1:
        out44 = out44.mean(axis=1)
    out = librosa.resample(out44, orig_sr=sr, target_sr=FS)
    out = out / np.max(np.abs(out))
    sf.write("results/voicefixer_out.wav", out, FS)

    clean, _ = sf.read("results/clean.wav")
    ref, out_aligned = align(clean, out)
    print(f"STOI voicefixer: {stoi_fn(ref, out_aligned, FS):.3f}")

    rev, _ = sf.read("results/reverberant.wav")
    ref_r, rev_a = align(clean, rev)
    print(f"STOI reverberant (unchanged): {stoi_fn(ref_r, rev_a, FS):.3f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wpe_out, _ = sf.read("results/dereverberated.wav")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, sig, title in zip(
        axes,
        [wpe_out, out],
        ["WPE (4 mics)", "voicefixer (1 mic)"],
    ):
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(sig, n_fft=512, hop_length=128)), ref=np.max
        )
        ax.imshow(
            D,
            origin="lower",
            aspect="auto",
            vmin=-80,
            vmax=0,
            cmap="magma",
            extent=[0, len(sig) / FS, 0, FS / 2000],
        )
        ax.set_title(title)
        ax.set_ylabel("kHz")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig("plots/wpe_vs_voicefixer.png", dpi=120)
    print("Wrote plots/wpe_vs_voicefixer.png")


if __name__ == "__main__":
    main()
