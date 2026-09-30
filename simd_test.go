package gtcrn

import (
	"math"
	"math/rand"
	"testing"
)

// The fast kernels against the generic ones, over the shapes the network uses
// and a few odd ones.

func TestGemmMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, sh := range [][3]int{{16, 24, 33}, {24, 8, 33}, {8, 45, 65}, {2, 48, 65}, {3, 5, 8}, {5, 7, 17}, {4, 3, 7}, {1, 1, 129}} {
		rows, inner, n := sh[0], sh[1], sh[2]
		os, xs := n+3, n+1
		w, x, b := randVec(rng, rows*inner), randVec(rng, inner*xs), randVec(rng, rows)
		want, got := randVec(rng, rows*os), make([]float32, rows*os)
		copy(got, want) // columns past n must survive
		gemmGeneric(want, os, w, x, xs, b, rows, inner, n)
		gemm(got, os, w, x, xs, b, rows, inner, n)
		for i := range want {
			if d := math.Abs(float64(want[i] - got[i])); d > 1e-5*(1+math.Abs(float64(want[i]))) {
				t.Fatalf("%v: element %d: %g, generic %g", sh, i, got[i], want[i])
			}
		}
	}
}

func TestDwconvMatchesGeneric(t *testing.T) {
	if !useFMA {
		t.Skip("no fast path")
	}
	rng := rand.New(rand.NewSource(3))
	for _, n := range []int{33, 8, 5, 40} {
		ch, xs, os := 16, n+3, n
		var x [3][]float32
		for i := range x {
			x[i] = randVec(rng, ch*xs)
		}
		w, b := randVec(rng, ch*9), randVec(rng, ch)
		got, want := make([]float32, ch*os), make([]float32, ch*os)
		dwconv(got, os, x, xs, w, b, ch, n)
		useFMA = false
		dwconv(want, os, x, xs, w, b, ch, n)
		useFMA = true
		for i := range want {
			if d := math.Abs(float64(want[i] - got[i])); d > 1e-5*(1+math.Abs(float64(want[i]))) {
				t.Fatalf("n=%d: element %d: %g, generic %g", n, i, got[i], want[i])
			}
		}
	}
}

func TestVecActivationsMatchScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, n := range []int{1, 7, 8, 9, 16, 33, 264, 528} {
		x := randVec(rng, n)
		for i := range x {
			x[i] *= 20
		}
		x[0] = -200 // clamped range
		s, th, pr := make([]float32, n), make([]float32, n), make([]float32, n)
		vecSigmoid(s, x)
		vecTanh(th, x)
		copy(pr, x)
		vecPReLU(pr, 0.25)
		for i, v := range x {
			if d := math.Abs(float64(s[i]) - 1/(1+math.Exp(-float64(v)))); d > 3e-7 {
				t.Errorf("n=%d sigmoid(%g) off by %.2g", n, v, d)
			}
			if d := math.Abs(float64(th[i]) - math.Tanh(float64(v))); d > 6e-7 {
				t.Errorf("n=%d tanh(%g) off by %.2g", n, v, d)
			}
			if want := max(v, 0) + 0.25*min(v, 0); pr[i] != want {
				t.Errorf("n=%d prelu(%g) = %g, want %g", n, v, pr[i], want)
			}
		}
	}
}

// TestGenericPathGolden runs the network goldens with the assembly off.
func TestGenericPathGolden(t *testing.T) {
	if !useFMA {
		t.Skip("generic path is already the one under test")
	}
	useFMA = false
	defer func() { useFMA = true }()
	TestNetworkGolden(t)
	TestDenoiserGolden(t)
}

func randVec(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}
