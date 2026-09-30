// Package gtcrn is a pure-Go port of GTCRN
// (https://github.com/Xiaobin-Rong/gtcrn), a 48k-parameter speech enhancement
// network, run as upstream's causal streaming model.
//
// The network was trained at 16 kHz on a 512-point STFT, 31.25 Hz per bin.
// At 8 and 12 kHz a 32 ms window gives the same bin spacing, so the spectrum
// is fed as is with the bins above the input's Nyquist left at zero: no
// resampling, and the same output quality as resampling to 16 kHz and back.
// Above 16 kHz there are no trained weights for the upper band, so those rates
// are refused.
package gtcrn

import (
	"errors"
	"fmt"
	"math"
)

// Options configures a Denoiser.
type Options struct {
	// SampleRate is 8000, 12000 or 16000.
	SampleRate int
	// Model is the weights to use; nil means DefaultModel.
	Model *Model
}

// Denoiser holds one stream's state. It is not safe for concurrent use.
type Denoiser struct {
	m          *Model
	n, hop, nb int
	scale      float32 // brings an n-point spectrum to the 512-point level
	win        []float32
	fft        *rfft
	spec       [bins]complex64
	frame      []float32
	in, ola    []float32
	re, im     [bins]float32 // network input, at the 512-point level
	xr, xi     [bins]float32 // the same spectrum unscaled
	mask       [2 * bins]float32
	st         *state
	s          *scratch
}

// New returns a Denoiser for one stream.
func New(o Options) (*Denoiser, error) {
	switch o.SampleRate {
	case 8000, 12000, 16000:
	default:
		return nil, fmt.Errorf("gtcrn: sample rate %d not supported; use 8000, 12000 or 16000", o.SampleRate)
	}
	m := o.Model
	if m == nil {
		var err error
		if m, err = DefaultModel(); err != nil {
			return nil, err
		}
	}
	n := o.SampleRate * 32 / 1000
	f, err := newRFFT(n)
	if err != nil {
		return nil, err
	}
	d := &Denoiser{
		m: m, n: n, hop: n / 2, nb: n/2 + 1,
		scale: 512 / float32(n),
		win:   make([]float32, n),
		fft:   f,
		frame: make([]float32, n),
		in:    make([]float32, n),
		ola:   make([]float32, n),
		st:    newState(),
		s:     &scratch{},
	}
	// Periodic sqrt-Hann on both sides; at 50% overlap the product sums to one.
	for i := range n {
		d.win[i] = float32(math.Sqrt(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n))))
	}
	return d, nil
}

// FrameSize is the number of samples per Process call: 16 ms.
func (d *Denoiser) FrameSize() int { return d.hop }

// Delay is the latency in samples: output sample i corresponds to input
// sample i-Delay().
func (d *Denoiser) Delay() int { return d.hop }

// Reset clears the stream state, as if newly created.
func (d *Denoiser) Reset() {
	clear(d.in)
	clear(d.ola)
	d.st.reset()
}

var errFrame = errors.New("gtcrn: in and out must both be FrameSize() samples")

// Process denoises one frame of samples scaled to [-1, 1]. in and out may be
// the same slice.
func (d *Denoiser) Process(out, in []float32) error {
	if len(in) != d.hop || len(out) != d.hop {
		return errFrame
	}
	n, hop := d.n, d.hop
	copy(d.in, d.in[hop:])
	copy(d.in[n-hop:], in)

	fr := d.frame
	for i, v := range d.in {
		fr[i] = v * d.win[i]
	}
	x := d.spec[:d.nb]
	d.fft.forward(x, fr)
	for k, v := range x {
		d.xr[k], d.xi[k] = real(v), imag(v)
		d.re[k], d.im[k] = d.xr[k]*d.scale, d.xi[k]*d.scale
	}
	d.m.forward(d.mask[:], d.re[:], d.im[:], d.st, d.s)

	for k := range x {
		mr, mi := d.mask[k], d.mask[bins+k]
		xr, xi := d.xr[k], d.xi[k]
		x[k] = complex(xr*mr-xi*mi, xi*mr+xr*mi)
	}
	d.fft.inverse(fr, x)
	for i, v := range fr {
		d.ola[i] += v * d.win[i]
	}
	copy(out, d.ola[:hop])
	copy(d.ola, d.ola[hop:])
	clear(d.ola[n-hop:])
	return nil
}

// ProcessInt16 is Process for 16-bit PCM.
func (d *Denoiser) ProcessInt16(out, in []int16) error {
	if len(in) != d.hop || len(out) != d.hop {
		return errFrame
	}
	buf := d.s.pcm[:d.hop]
	for i, v := range in {
		buf[i] = float32(v) / 32768
	}
	if err := d.Process(buf, buf); err != nil {
		return err
	}
	for i, v := range buf {
		v = float32(math.Round(float64(v) * 32768))
		out[i] = int16(max(-32768, min(32767, v)))
	}
	return nil
}
