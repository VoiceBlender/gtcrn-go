package gtcrn

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"
)

// The golden files come from upstream's PyTorch streaming model via
// tools/export. Agreement is bounded rather than exact: float32 summation
// order and the exp approximation differ from torch.

func readF32(t *testing.T, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

func readI16(t *testing.T, path string) []int16 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	v := make([]int16, len(b)/2)
	for i := range v {
		v[i] = int16(binary.LittleEndian.Uint16(b[2*i:]))
	}
	return v
}

// snrDB is the reference's energy over the error's.
func snrDB(ref, got []float32) float64 {
	var s, e float64
	for i := range ref {
		d := float64(ref[i]) - float64(got[i])
		s += float64(ref[i]) * float64(ref[i])
		e += d * d
	}
	return 10 * math.Log10(s/e)
}

func mustModel(t testing.TB) *Model {
	m, err := DefaultModel()
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestNetworkGolden(t *testing.T) {
	m := mustModel(t)
	in := readF32(t, "testdata/net_in.bin")
	want := readF32(t, "testdata/net_out.bin")
	frames := len(in) / (2 * bins)
	st, s := newState(), &scratch{}
	var re, im [bins]float32
	var mask [2 * bins]float32
	got := make([]float32, len(want))
	for f := range frames {
		x := in[f*2*bins : (f+1)*2*bins]
		for k := range bins {
			re[k], im[k] = x[2*k], x[2*k+1]
		}
		m.forward(mask[:], re[:], im[:], st, s)
		for k := range bins {
			mr, mi := mask[k], mask[bins+k]
			got[f*2*bins+2*k] = re[k]*mr - im[k]*mi
			got[f*2*bins+2*k+1] = im[k]*mr + re[k]*mi
		}
	}
	for f := range frames {
		lo, hi := f*2*bins, (f+1)*2*bins
		if snr := snrDB(want[lo:hi], got[lo:hi]); snr < 100 {
			t.Errorf("frame %d: %.1f dB against torch, want >= 100", f, snr)
		}
	}
	snr := snrDB(want, got)
	t.Logf("all frames: %.1f dB", snr)
	if snr < 120 {
		t.Errorf("all frames: %.1f dB against torch, want >= 120", snr)
	}
}

func TestDenoiserGolden(t *testing.T) {
	for _, rate := range []int{8000, 12000, 16000} {
		t.Run(fmt.Sprint(rate), func(t *testing.T) {
			in := readI16(t, fmt.Sprintf("testdata/in_%d.bin", rate))
			want := readF32(t, fmt.Sprintf("testdata/out_%d.bin", rate))
			d, err := New(Options{SampleRate: rate})
			if err != nil {
				t.Fatal(err)
			}
			hop := d.FrameSize()
			got := make([]float32, len(want))
			buf := make([]float32, hop)
			for i := 0; i+hop <= len(want); i += hop {
				for j := range buf {
					buf[j] = float32(in[i+j]) / 32768
				}
				if err := d.Process(got[i:i+hop], buf); err != nil {
					t.Fatal(err)
				}
			}
			snr := snrDB(want, got)
			t.Logf("%.1f dB against torch", snr)
			if snr < 110 {
				t.Errorf("%.1f dB against torch, want >= 110", snr)
			}
		})
	}
}

func TestResetMatchesNew(t *testing.T) {
	in := readI16(t, "testdata/in_16000.bin")
	d, _ := New(Options{SampleRate: 16000})
	hop := d.FrameSize()
	run := func() []int16 {
		out := make([]int16, 20*hop)
		for i := 0; i < len(out); i += hop {
			if err := d.ProcessInt16(out[i:i+hop], in[i:i+hop]); err != nil {
				t.Fatal(err)
			}
		}
		return out
	}
	a := run()
	d.Reset()
	b := run()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("sample %d: %d after Reset, %d fresh", i, b[i], a[i])
		}
	}
}

func TestRejectsRate(t *testing.T) {
	for _, r := range []int{0, 11025, 24000, 48000} {
		if _, err := New(Options{SampleRate: r}); err == nil {
			t.Errorf("rate %d accepted", r)
		}
	}
}

func TestExpf(t *testing.T) {
	var worst float64
	for x := float32(-80); x <= 80; x += 0.0137 {
		want := math.Exp(float64(x))
		if e := math.Abs(float64(expf(x))-want) / want; e > worst {
			worst = e
		}
	}
	if worst > 5e-7 {
		t.Errorf("expf relative error %.2g, want <= 5e-7", worst)
	}
}

func TestFFT(t *testing.T) {
	for _, n := range []int{256, 384, 512} {
		f, err := newFFT(n)
		if err != nil {
			t.Fatal(err)
		}
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(math.Sin(float64(i*i)*0.37), math.Cos(float64(i)*1.3))
			f.a[i] = complex64(x[i])
		}
		y := f.transform()
		var worst float64
		for k := range n {
			var acc complex128
			for i, v := range x {
				a := -2 * math.Pi * float64(i*k%n) / float64(n)
				acc += v * complex(math.Cos(a), math.Sin(a))
			}
			d := acc - complex128(y[k])
			worst = max(worst, math.Hypot(real(d), imag(d)))
		}
		if worst > 1e-3 {
			t.Errorf("n=%d: max error %.2g against a direct DFT", n, worst)
		}
	}
}

func TestRFFT(t *testing.T) {
	for _, n := range []int{256, 384, 512} {
		r, err := newRFFT(n)
		if err != nil {
			t.Fatal(err)
		}
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(math.Sin(float64(i*i) * 0.37))
		}
		X := make([]complex64, n/2+1)
		r.forward(X, x)
		var worst float64
		for k := range X {
			var acc complex128
			for i, v := range x {
				a := -2 * math.Pi * float64(i*k%n) / float64(n)
				acc += complex(float64(v), 0) * complex(math.Cos(a), math.Sin(a))
			}
			d := acc - complex128(X[k])
			worst = max(worst, math.Hypot(real(d), imag(d)))
		}
		if worst > 1e-3 {
			t.Errorf("n=%d: forward off by %.2g", n, worst)
		}
		y := make([]float32, n)
		r.inverse(y, X)
		for i := range x {
			if d := math.Abs(float64(y[i] - x[i])); d > 1e-5 {
				t.Fatalf("n=%d: round trip sample %d off by %.2g", n, i, d)
			}
		}
	}
}
