# Blind Dereverberation — WPE vs. Neural Restoration

Hands-on comparison of speech dereverberation approaches on the same simulated
reverberant recording, from classical to neural:

- **WPE** (Weighted Prediction Error) via
  [nara_wpe](https://github.com/fgnt/nara_wpe) — the classical blind method:
  no room knowledge, no trained model, just a long-term linear prediction
  filter estimated from the observed audio itself.
- **voicefixer** — a pretrained neural restoration model that re-synthesizes
  clean speech through a neural vocoder from a single microphone.

Both are evaluated with STOI against the clean reference, with before/after
audio and spectrograms. WPE (4 mics) reaches STOI 0.852 and voicefixer (1 mic)
0.863, from a reverberant baseline of 0.645 — with very different tradeoffs,
explained below.

## Project landscape

Survey of open-source blind dereverberation projects, ordered from simplest to
most advanced. This repo implements demos from Tier 1 (nara_wpe) and Tier 2
(voicefixer).

### Tier 1 — Simple classical (statistical, no training, CPU-only)

| Project | Stars | Notes |
|---|---|---|
| [fgnt/nara_wpe](https://github.com/fgnt/nara_wpe) | 566 | WPE — *the* standard blind dereverberation baseline. `pip install nara-wpe`, offline/block-online/frame-online variants, single- or multi-channel, example notebooks with audio. MIT. |
| [helianvine/fdndlp](https://github.com/helianvine/fdndlp) | 158 | Same WPE family (variance-normalized delayed linear prediction), MATLAB + Python — good for studying the math. |
| [Debapriya-Tula/Speech_Dereverberation](https://github.com/Debapriya-Tula/Speech_Dereverberation), [mrajeswarasai/Speech-Dereverberation](https://github.com/mrajeswarasai/Speech-Dereverberation) | small | Minimal student-scale WPE implementations — easiest code to read end-to-end. |

### Tier 2 — Pretrained deep learning (download checkpoint, run inference)

| Project | Stars | Notes |
|---|---|---|
| [haoheliu/voicefixer](https://github.com/haoheliu/voicefixer) | 1.4k | General speech restoration (dereverb + denoise + declip + bandwidth) with pretrained models and a simple API — the easiest "it just works" neural demo. |
| [DiegoLeon96/Neural-Speech-Dereverberation](https://github.com/DiegoLeon96/Neural-Speech-Dereverberation) | 120 | U-Net-style models, good mid-complexity learning repo. |
| [MathWorks deep-learning example](https://www.mathworks.com/help/audio/ug/dereverberate-speech-using-deep-learning-networks.html) | — | Pretrained U-Net walkthrough (MATLAB). |

### Tier 3 — Research-grade blind/unsupervised (GPU, diffusion/VAE methods)

| Project | Notes |
|---|---|
| [sp-uhh/buddy](https://github.com/sp-uhh/buddy) | "Single-Channel **Blind** Unsupervised Dereverberation with Diffusion Models." Pretrained VCTK checkpoint; closest match to modern "blind dereverberation" literature. |
| [Audio-WestlakeU/RVAE-EM](https://github.com/Audio-WestlakeU/RVAE-EM) | Recurrent VAE + EM (ICASSP 2024), unsupervised and supervised variants. |
| [Audio-WestlakeU/VINP](https://github.com/Audio-WestlakeU/VINP) | Joint dereverberation + blind RIR identification (IEEE TASLP). |
| [rrbluke/BSSD](https://github.com/rrbluke/BSSD) | Joint blind source separation + dereverberation. |

Listening demos (no install): [Sony unsupervised vocal dereverberation](https://koichi-saito-sony.github.io/unsupervised-vocal-dereverb/audio_samples.html) ·
[PEVD speech dereverberation](https://vwn09.github.io/pevd-enhance/dereverb.html) ·
[USD-DPS multi-channel diffusion](https://arxiv.org/html/2508.02071v1)

## Setup (WPE demo)

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

### How voicefixer works

VoiceFixer (Liu et al., 2021: *"VoiceFixer: Toward General Speech Restoration
with Neural Vocoder"*) targets **General Speech Restoration** — one model that
simultaneously fixes noise, reverberation, clipping, and low bandwidth, rather
than one specialized model per distortion. It never needs to be told which
degradation is present.

**Two-stage architecture:**

1. **Analysis** — a ResUNet operates on the **mel spectrogram** of the degraded
   input and predicts the clean mel spectrogram. Mel is compact and perceptually
   weighted, and discards phase/fine detail, so this stage only has to get the
   perceptual "sketch" of clean speech right. (This is the ~467 MB `vf.ckpt`.)
2. **Synthesis** — a TFGAN-based **neural vocoder**, pretrained separately on
   clean 44.1 kHz speech, generates a brand-new waveform from that mel sketch.
   (This is the ~135 MB vocoder checkpoint.)

Nothing of the original waveform survives to the output — not phase, not sample
values. This is the opposite of WPE's `output = input − predicted_late_reverb`,
which can only ever remove energy from the real signal. Consequences:

- **Crisp gaps**: the vocoder generates nothing where the estimated mel says
  silence, while WPE leaves whatever residual its linear filter couldn't predict.
- **Always 44.1 kHz out**: the analysis stage learned bandwidth extension, so it
  *invents* plausible content above the input's Nyquist — useful for restoring
  old recordings, but fabricated detail (the demo resamples back to 16 kHz
  before scoring STOI).
- **Hallucination risk**: if analysis misreads a degraded phone or word, the
  vocoder confidently synthesizes the *wrong* clean speech. No mechanism forces
  fidelity to the input the way WPE's subtractive math does.
- **Timbre drift**: the voice is re-generated, so speaker identity can shift
  slightly.

**Training**: the analysis stage is trained supervised on *simulated*
degradations — clean 44.1 kHz speech (VCTK and others) convolved with RIRs,
mixed with noise, clipped, and band-limited, in random combinations. So it is
supervised in training but still **blind at inference**: it learned the
statistical signature of each distortion family, not any specific room. The
vocoder is trained separately on clean speech with GAN + spectral losses.

**API**: `VoiceFixer().restore(input, output, cuda, mode)` with `mode=0` the
standard path (used here), `mode=1` adding input normalization for very
quiet/loud recordings, `mode=2` a try-everything convenience. A separate
`Vocoder` class accepts your own mel spectrograms.

### Placing the methods

WPE is transparent, provably non-destructive, multi-channel-aware, and instant —
the safe default and standard ASR front-end. VoiceFixer trades those guarantees
for generative power: single-channel, compound degradations, bandwidth
restoration, subjectively cleaner audio — but it repairs by resynthesis and can
hallucinate. BUDDy (next rung, not included here) keeps the generative power but
is *unsupervised*: no paired training data, jointly estimating the reverb
operator and clean speech via diffusion posterior sampling.

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
