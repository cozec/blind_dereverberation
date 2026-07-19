# Blind Dereverberation — WPE Demo

Minimal demo of blind speech dereverberation using the classical **Weighted
Prediction Error (WPE)** algorithm via [nara_wpe](https://github.com/fgnt/nara_wpe).
"Blind" means no knowledge of the room impulse response and no trained model —
WPE estimates a long-term linear prediction filter directly from the observed
reverberant audio.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install nara-wpe soundfile matplotlib pystoi
```

## Run

```bash
python src/wpe_demo.py
```

The script is self-contained — on first run it downloads the four clean CMU
Arctic utterances into `data/`. It then:
1. Concatenates the utterances (16 kHz mono, ~12.6 s).
2. Simulates a 4-microphone reverberant recording (synthetic RIRs: unit direct
   path + exponentially decaying noise tail, RT60 = 0.6 s).
3. Runs offline multi-channel WPE (taps=10, delay=3, 5 iterations) in the STFT domain.
4. Writes `results/clean.wav`, `results/reverberant.wav`, `results/dereverberated.wav`
   and `plots/spectrograms.png`, and prints STOI intelligibility scores.

## Result

| Signal | Mics | STOI (vs clean) |
|---|---|---|
| Reverberant (mic 0, RT60 0.6 s) | — | 0.645 |
| WPE dereverberated | 4 | 0.852 |
| voicefixer restored | 1 | **0.863** |

WPE runs in a few seconds on CPU. Listen to the WAVs in `results/` for the
before/after comparison.

## Neural comparison: voicefixer

[voicefixer](https://github.com/haoheliu/voicefixer) is a pretrained neural
speech-restoration model (analysis network + neural vocoder that *resynthesizes*
the waveform). It needs Python ≤3.11 for its torch dependency, so it lives in a
separate venv:

```bash
python3.11 -m venv .venv-vf
.venv-vf/bin/pip install voicefixer pystoi
.venv-vf/bin/python src/voicefixer_demo.py   # run src/wpe_demo.py first
```

On first use it downloads ~630 MB of checkpoints to `~/.cache/voicefixer/`
(if the built-in Zenodo download stalls, fetch
`https://zenodo.org/record/5600188/files/vf.ckpt` manually with
`curl -L -C -` into `~/.cache/voicefixer/analysis_module/checkpoints/`).

From a **single** microphone it slightly beats 4-mic WPE on this clip
(STOI 0.863 vs 0.852, `plots/wpe_vs_voicefixer.png`), with noticeably crisper
silence gaps — but unlike WPE it resynthesizes audio through a vocoder, so it
can alter timbre/content, and it needs ~1 min on CPU vs seconds for WPE.

## How WPE / nara_wpe works

`nara_wpe` is the reference implementation of **Weighted Prediction Error (WPE)**
from the University of Paderborn (fgnt), based on the NTT papers by Nakatani,
Yoshioka et al. ("NARA" nods to NTT's lab in Nara, Japan). It was a key ASR
front-end in winning CHiME challenge systems.

### Core idea

Reverberation is a convolution: the mic signal is clean speech smeared over time
by the room impulse response. WPE exploits the fact that **late reverberation is
predictable from the signal's own past**, while clean speech is only weakly
correlated with itself at long lags. In the STFT domain, per frequency band, it
fits a *delayed* linear prediction filter:

```
late reverb estimate at frame t = G^H · [ y(t−Δ), …, y(t−Δ−K+1) ]
dereverberated x(t) = y(t) − late reverb estimate
```

The two filter parameters map directly to the demo's arguments:

- **`delay` (Δ = 3 frames)** — the predictor skips the most recent frames, so the
  direct sound and early reflections (which aid intelligibility) are left
  untouched. Only energy predictable from further back — the late tail — is
  subtracted. Without the delay the filter would whiten the speech itself.
- **`taps` (K = 10 frames)** — how far back the predictor looks, i.e. how long a
  reverb tail it can model (~80 ms per step with a 512/128 STFT at 16 kHz).

### The "weighted" part

Plain least-squares prediction would let loud speech frames dominate and eat the
speech. WPE instead models clean speech at each time-frequency point as Gaussian
with time-varying variance λ(t,f) and minimizes the prediction error **weighted
by 1/λ**. Since λ is unknown, it alternates (the `iterations=5` argument):

1. Solve for filter coefficients G given current λ (weighted least squares).
2. Dereverberate, re-estimate λ from the output power.
3. Repeat — converges in 3–5 iterations.

This is maximum-likelihood estimation and fully **blind**: no room knowledge, no
training — the filter is estimated from the utterance being processed.

### Multi-channel and API notes

With multiple mics, WPE predicts each channel's late reverb from the past of
*all* channels jointly, which constrains the estimate much better (why the 4-mic
demo works so well). Single-channel works too, just less powerfully. The output
keeps the input channel count — WPE removes reverb, it does not beamform.

API gotcha: `wpe()` expects `(frequency, channel, time)`, while the package's
`stft` utility returns `(channel, time, frequency)` — hence the
`transpose(2, 0, 1)` in [src/wpe_demo.py](src/wpe_demo.py).

The package ships offline batch WPE (best quality, used here), block-online WPE
for streaming, a recursive frame-online `OnlineWPE` (RLS-style, real-time), and
parallel NumPy and TensorFlow implementations.

### Strengths and limits

Blind, no training data, no GPU, transparent math, and artifact-free — it only
subtracts a predicted linear component, so it cannot hallucinate content. That is
why it remains the default ASR front-end even in the deep-learning era. Limits:
it does not model additive noise, offline quality needs a few seconds of audio
for stable statistics (why the demo concatenates four utterances), and a linear
per-frequency filter cannot fully invert a real room — learned methods go
further at the cost of data, GPUs, and possible hallucination.

## Going further

- Pretrained neural restoration: [voicefixer](https://github.com/haoheliu/voicefixer)
- Blind unsupervised diffusion (single-channel, SOTA): [BUDDy](https://github.com/sp-uhh/buddy)
