"""Prepare our demo clip for BUDDy's VCTKTestPaired dataset layout.

Writes buddy/my_data/clean/adam/adam_001.wav (first 4 s of the clean demo
signal) and buddy/my_data/rir/adam/adam_001.wav (the mic-0 synthetic RIR,
regenerated with the same seed as src/wpe_demo.py). BUDDy's tester convolves
clean * RIR itself and runs blind dereverberation on the result.
"""

import os

import numpy as np
import soundfile as sf

from wpe_demo import FS, RT60, RNG, UTTERANCES, make_rir

SEG = 4 * FS

clean = np.concatenate([sf.read(f"data/{f}.wav")[0] for f in UTTERANCES])
clean = clean / np.max(np.abs(clean))
rir = make_rir(RT60, FS, RNG)  # first draw from seed-0 RNG == mic 0 in wpe_demo

os.makedirs("buddy/my_data/clean/adam", exist_ok=True)
os.makedirs("buddy/my_data/rir/adam", exist_ok=True)
sf.write("buddy/my_data/clean/adam/adam_001.wav", clean[:SEG], FS)
sf.write("buddy/my_data/rir/adam/adam_001.wav", rir, FS)
print("Wrote buddy/my_data (clean 4 s + mic-0 RIR)")
