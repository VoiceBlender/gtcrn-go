package gtcrn

import (
	"fmt"
	"math"
)

// fft is an unscaled forward complex FFT for sizes 2^a * 3^b, as a Stockham
// autosort: no bit reversal, radix 4 where it can.
type fft struct {
	n      int
	stages []fftStage
	a, b   []complex64
}

type fftStage struct {
	p, m, s int
	tw      []complex64 // tw[j*p+r] = exp(-2πi j r / (p*m))
}

func newFFT(n int) (*fft, error) {
	f := &fft{n: n, a: make([]complex64, n), b: make([]complex64, n)}
	rest, s := n, 1
	for rest > 1 {
		var p int
		switch {
		case rest%4 == 0:
			p = 4
		case rest%2 == 0:
			p = 2
		case rest%3 == 0:
			p = 3
		default:
			return nil, fmt.Errorf("gtcrn: no FFT for size %d", n)
		}
		m := rest / p
		st := fftStage{p: p, m: m, s: s, tw: make([]complex64, m*p)}
		for j := range m {
			for r := range p {
				a := -2 * math.Pi * float64(j*r) / float64(rest)
				st.tw[j*p+r] = complex(float32(math.Cos(a)), float32(math.Sin(a)))
			}
		}
		f.stages = append(f.stages, st)
		rest, s = m, s*p
	}
	return f, nil
}

// transform reads f.a and returns the slice holding the result, which is one
// of f's own buffers.
func (f *fft) transform() []complex64 {
	x, y := f.a, f.b
	const h = 0.866025404 // sin(π/3)
	for _, st := range f.stages {
		m, s := st.m, st.s
		switch st.p {
		case 4:
			for j := range m {
				w1, w2, w3 := st.tw[4*j+1], st.tw[4*j+2], st.tw[4*j+3]
				for q := range s {
					a0, a1 := x[q+s*j], x[q+s*(j+m)]
					a2, a3 := x[q+s*(j+2*m)], x[q+s*(j+3*m)]
					t0, t1 := a0+a2, a0-a2
					t2, t3 := a1+a3, a1-a3
					t3 = complex(imag(t3), -real(t3)) // -i
					o := q + s*4*j
					y[o] = t0 + t2
					y[o+s] = (t1 + t3) * w1
					y[o+2*s] = (t0 - t2) * w2
					y[o+3*s] = (t1 - t3) * w3
				}
			}
		case 2:
			for j := range m {
				w := st.tw[2*j+1]
				for q := range s {
					a, b := x[q+s*j], x[q+s*(j+m)]
					o := q + s*2*j
					y[o] = a + b
					y[o+s] = (a - b) * w
				}
			}
		case 3:
			for j := range m {
				w1, w2 := st.tw[3*j+1], st.tw[3*j+2]
				for q := range s {
					a0, a1, a2 := x[q+s*j], x[q+s*(j+m)], x[q+s*(j+2*m)]
					t := a1 + a2
					d := a1 - a2
					c := a0 - t*0.5
					rd := complex(h*imag(d), -h*real(d)) // -i·sin(π/3)·d
					o := q + s*3*j
					y[o] = a0 + t
					y[o+s] = (c + rd) * w1
					y[o+2*s] = (c - rd) * w2
				}
			}
		}
		x, y = y, x
	}
	return x
}

// rfft is the real-input FFT of even size n, through a complex FFT of n/2.
type rfft struct {
	n int
	c *fft
	w []complex64 // exp(-2πi k/n), k < n/2
}

func newRFFT(n int) (*rfft, error) {
	c, err := newFFT(n / 2)
	if err != nil {
		return nil, err
	}
	r := &rfft{n: n, c: c, w: make([]complex64, n/2)}
	for k := range r.w {
		a := -2 * math.Pi * float64(k) / float64(n)
		r.w[k] = complex(float32(math.Cos(a)), float32(math.Sin(a)))
	}
	return r, nil
}

// forward sets X[0..n/2] to the unscaled DFT of x.
func (r *rfft) forward(X []complex64, x []float32) {
	h := r.n / 2
	a := r.c.a
	for m := range h {
		a[m] = complex(x[2*m], x[2*m+1])
	}
	z := r.c.transform()
	X[0] = complex(real(z[0])+imag(z[0]), 0)
	X[h] = complex(real(z[0])-imag(z[0]), 0)
	for k := 1; k < h; k++ {
		zk, zc := z[k], z[h-k]
		zc = complex(real(zc), -imag(zc))
		e := (zk + zc) * 0.5
		o := (zk - zc) * complex(0, -0.5)
		X[k] = e + r.w[k]*o
	}
}

// inverse sets x to the inverse DFT, 1/n scaled, of the Hermitian spectrum
// whose first half is X[0..n/2]. The imaginary parts of X[0] and X[n/2] are
// ignored, as a real inverse does.
func (r *rfft) inverse(x []float32, X []complex64) {
	h := r.n / 2
	a := r.c.a
	x0, xh := real(X[0]), real(X[h])
	// Z = E + iO with E, O the spectra of the even and odd samples; the
	// transform runs on conj(Z) so a forward FFT inverts it.
	a[0] = complex((x0+xh)*0.5, -(x0-xh)*0.5)
	for k := 1; k < h; k++ {
		xk, xc := X[k], X[h-k]
		xc = complex(real(xc), -imag(xc))
		e := (xk + xc) * 0.5
		w := r.w[k]
		o := (xk - xc) * complex(real(w), -imag(w)) * 0.5
		z := e + complex(-imag(o), real(o))
		a[k] = complex(real(z), -imag(z))
	}
	t := r.c.transform()
	g := 1 / float32(h)
	for m := range h {
		x[2*m] = real(t[m]) * g
		x[2*m+1] = -imag(t[m]) * g
	}
}
