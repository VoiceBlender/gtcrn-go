"""Convert upstream GTCRN weights to weights.bin and regenerate the golden files.

Needs torch, numpy, scipy and soundfile, plus a checkout of
https://github.com/Xiaobin-Rong/gtcrn at the revision in MODEL_VERSION.

    python tools/export/export.py /path/to/gtcrn

Blob format, little-endian: "GTCRNw01", u32 tensor count, then per tensor
u16 name length, name, u8 rank, u32 dims, float32 data. Tensors are the
streaming model's state dict as convert_to_stream leaves it, BatchNorm
unfolded; the Go loader does the folding.
"""
import importlib.util, os, struct, sys

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

up = os.path.abspath(sys.argv[1])
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.normpath(os.path.join(here, '..', '..'))
sys.path.insert(0, os.path.join(up, 'stream'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


gt = load('gtcrn_offline', os.path.join(up, 'gtcrn.py'))
st = load('gtcrn_stream', os.path.join(up, 'stream', 'gtcrn_stream.py'))
from modules.convert import convert_to_stream  # noqa: E402

# checkpoints/ scores higher than the copy under stream/onnx_models/ on our
# corpus, at both 8 and 16 kHz.
offline = gt.GTCRN().eval()
offline.load_state_dict(torch.load(os.path.join(up, 'checkpoints', 'model_trained_on_dns3.tar'),
                                   map_location='cpu')['model'])
model = st.StreamGTCRN().eval()
convert_to_stream(model, offline)

tensors = [(k, v.detach().numpy().astype('<f4')) for k, v in model.state_dict().items()
           if not k.endswith('num_batches_tracked')]
with open(os.path.join(root, 'weights.bin'), 'wb') as f:
    f.write(b'GTCRNw01' + struct.pack('<I', len(tensors)))
    for k, a in tensors:
        n = k.encode()
        f.write(struct.pack('<H', len(n)) + n + struct.pack('<B', a.ndim))
        f.write(struct.pack(f'<{a.ndim}I', *a.shape) + a.tobytes())


class Stream:
    """The framing gtcrn.go uses: sqrt-Hann, 32 ms window, 50% overlap, bins
    above the input's Nyquist zeroed, spectrum scaled to the 512-point level."""

    def __init__(self, rate):
        self.n = rate * 32 // 1000
        self.h = self.n // 2
        self.nb = self.n // 2 + 1
        self.scale = 512 / self.n
        self.win = np.sqrt(np.hanning(self.n + 1)[:-1])
        self.inbuf = np.zeros(self.n)
        self.ola = np.zeros(self.n)
        self.cc = torch.zeros(2, 1, 16, 16, 33)
        self.tc = torch.zeros(2, 3, 1, 1, 16)
        self.ic = torch.zeros(2, 1, 33, 16)

    def net(self, spec):
        """spec: complex (257,) at the 512-point level. Returns the enhanced spectrum."""
        x = torch.from_numpy(np.stack([spec.real, spec.imag], -1).astype(np.float32)).reshape(1, 257, 1, 2)
        with torch.no_grad():
            y, self.cc, self.tc, self.ic = model(x, self.cc, self.tc, self.ic)
        y = y.numpy()[0, :, 0]
        return y[:, 0] + 1j * y[:, 1]

    def process(self, frame):
        self.inbuf = np.concatenate([self.inbuf[self.h:], frame])
        spec = np.zeros(257, complex)
        spec[:self.nb] = np.fft.rfft(self.inbuf * self.win) * self.scale
        y = self.net(spec)[:self.nb] / self.scale
        self.ola += np.fft.irfft(y, self.n) * self.win
        out = self.ola[:self.h].copy()
        self.ola = np.concatenate([self.ola[self.h:], np.zeros(self.h)])
        return out


mix, rate = sf.read(os.path.join(up, 'test_wavs', 'mix.wav'), dtype='int16')
assert rate == 16000
td = os.path.join(root, 'testdata')
os.makedirs(td, exist_ok=True)

# Network golden: 60 frames, enough for the dilation-5 caches to fill several times.
s = Stream(16000)
x = mix[16000:16000 + 61 * s.h].astype(np.float64) / 32768
ins, outs = [], []
buf = np.zeros(s.n)
for i in range(60):
    buf = np.concatenate([buf[s.h:], x[i * s.h:(i + 1) * s.h]])
    spec = np.fft.rfft(buf * s.win)
    ins.append(spec)
    outs.append(s.net(spec))
for name, a in [('net_in', ins), ('net_out', outs)]:
    a = np.array(a)
    np.stack([a.real, a.imag], -1).astype('<f4').tofile(os.path.join(td, name + '.bin'))

# Denoiser goldens: 2 s at each supported rate.
for r in (8000, 12000, 16000):
    x = mix[16000:16000 * 3]
    if r != 16000:
        g = np.gcd(r, 16000)
        x = np.clip(np.round(resample_poly(x.astype(np.float64), r // g, 16000 // g)), -32768, 32767).astype(np.int16)
    s = Stream(r)
    y = np.concatenate([s.process(x[i:i + s.h].astype(np.float64) / 32768) for i in range(0, len(x) - s.h + 1, s.h)])
    x[:len(y)].astype('<i2').tofile(os.path.join(td, f'in_{r}.bin'))
    y.astype('<f4').tofile(os.path.join(td, f'out_{r}.bin'))

print('wrote weights.bin and testdata/')
