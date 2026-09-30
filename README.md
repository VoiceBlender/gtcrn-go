# gtcrn-go

Pure-Go port of [GTCRN](https://github.com/Xiaobin-Rong/gtcrn), a 48k-parameter
speech enhancement network, run as upstream's causal streaming model. No cgo;
one dependency, `golang.org/x/sys`, for CPU feature detection. The 206 KB of
weights are embedded.

## Usage

```go
import gtcrn "github.com/VoiceBlender/gtcrn-go"

d, err := gtcrn.New(gtcrn.Options{SampleRate: 8000})
// d.FrameSize() == 128 samples == 16 ms
for /* each frame */ {
    err := d.ProcessInt16(out, in)
}
```

`Process` takes float samples in [-1, 1]. Output lags input by `Delay()`
samples, one frame. Allocation-free after `New`. A `Model` is read-only and
shared by any number of Denoisers; each stream needs its own `Denoiser`.

## Sample rates

8, 12 and 16 kHz, with no resampling. The network was trained on a 16 kHz,
512-point STFT; a 32 ms window keeps the same 31.25 Hz bin spacing at the lower
rates, and the bins above Nyquist are fed as zeros, which is what an upsampled
signal would give it anyway. Measured on 8 kHz telephony, this matches resampling
to 16 kHz and back to within 0.02 PESQ. Above 16 kHz there are no trained
weights for the upper band, so those rates are refused.

## Accuracy

Agrees with upstream's PyTorch streaming model to about 130 dB SNR, at the
network and end to end at every rate. The goldens in `testdata/` come from
`make model`.

## Performance

One core of a Ryzen 9 7900, per 10 ms of audio. Sessions per CPU is measured
with all 24 hardware threads busy.

| | µs/10 ms | ×RT | sessions/CPU |
|---|---|---|---|
| gtcrn-go, 8 kHz | 24 | 410 | ~4000 |
| gtcrn-go, 16 kHz | 26 | 385 | ~3900 |
| rnnoise-go, 8 kHz | 55 | 181 | 1712 |
| rnnoise-go, 16 kHz | 64 | 157 | 1484 |

On amd64 with AVX2 and FMA, the matrix products, the depthwise conv and the
activations run in assembly (`simd_amd64.s`). Elsewhere the portable Go path
takes 76 to 86 µs; `BenchmarkProcessGeneric` measures it. Cost is nearly flat
across rates because the network dominates.

## Quality against rnnoise-go

Ten speech clips (eight 8 kHz telephony, two 16 kHz) mixed with call-centre and
street noise at 0, 5 and 10 dB SNR:

| | PESQ | STOI | SI-SDR |
|---|---|---|---|
| 16 kHz, rnnoise-go | 1.63 | 0.928 | 9.1 dB |
| 16 kHz, gtcrn-go | 1.79 | 0.923 | 11.1 dB |
| 8 kHz, rnnoise-go | 2.63 | 0.930 | 9.6 dB |
| 8 kHz, gtcrn-go | 2.66 | 0.922 | 11.3 dB |

Wideband PESQ at 16 kHz, narrowband at 8 kHz. The corpus is small and mostly
band-limited to 4 kHz.

## Command line

```
go install github.com/VoiceBlender/gtcrn-go/cmd/gtcrn@latest
gtcrn noisy.wav clean.wav
```

16-bit PCM WAV at 8, 12 or 16 kHz; the delay is compensated.

## Licence

MIT, upstream's terms; see `LICENSE` and `NOTICE` for provenance.
