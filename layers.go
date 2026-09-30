package gtcrn

import "math"

// Activations are stored channel-major, [channel][frequency], one frame at a
// time: the streaming model never sees more than one time step. Every
// BatchNorm is folded into the layer before it at load time.

const (
	actNone = iota
	actPReLU
	actTanh
)

// loadBN returns BatchNorm's inference-mode per-channel scale and shift.
func loadBN(l *loader, p string, n int) (scale, shift []float32) {
	g, b := l.get(p+".weight", n), l.get(p+".bias", n)
	mean, v := l.get(p+".running_mean", n), l.get(p+".running_var", n)
	scale, shift = make([]float32, n), make([]float32, n)
	if l.err != nil {
		return
	}
	for i := range n {
		s := float64(g[i]) / math.Sqrt(float64(v[i])+1e-5)
		scale[i] = float32(s)
		shift[i] = float32(float64(b[i]) - float64(mean[i])*s)
	}
	return
}

// foldBN applies BatchNorm to weights stored [out][...] and, if b is not nil,
// to the bias.
func foldBN(w, b, scale, shift []float32) {
	if w == nil {
		return
	}
	per := len(w) / len(scale)
	for o, s := range scale {
		for i := range per {
			w[o*per+i] *= s
		}
		if b != nil {
			b[o] = b[o]*s + shift[o]
		}
	}
}

func loadAlpha(l *loader, name string) float32 {
	if a := l.get(name, 1); a != nil {
		return a[0]
	}
	return 0
}

func activate(x []float32, act int, alpha float32) {
	switch act {
	case actPReLU:
		vecPReLU(x, alpha)
	case actTanh:
		vecTanh(x, x)
	}
}

// fillRows sets each of the len(b) rows of width n to its bias.
func fillRows(dst, b []float32, n int) {
	for o, v := range b {
		row := dst[o*n : (o+1)*n]
		for i := range row {
			row[i] = v
		}
	}
}

// convF is a Conv2d or ConvTranspose2d with a (1,5) kernel, stride 2 and
// padding 2 along frequency, then BatchNorm and an activation. Both run as
// matrix products over rows of shifted input.
type convF struct {
	in, out, groups int
	transposed      bool
	// Forward: [out][in/g][5 taps]. Transposed: [out][in/g][3 taps] for the
	// even outputs, which read x[e-1], x[e], x[e+1], and wOdd likewise for the
	// odd outputs, whose first tap is zero.
	w, wOdd []float32
	b       []float32
	act     int
	alpha   float32
}

func loadConvF(l *loader, p string, in, out, groups int, transposed bool, act int) convF {
	c := convF{in: in, out: out, groups: groups, transposed: transposed, act: act}
	ig, og := in/groups, out/groups
	c.b = l.get(p+".conv.bias", out)
	scale, shift := loadBN(l, p+".bn", out)
	if transposed {
		// Output 2e-2+k takes input e through tap k.
		tw := l.get(p+".conv.weight", in, og, 1, 5)
		if tw != nil {
			c.w, c.wOdd = make([]float32, out*ig*3), make([]float32, out*ig*3)
			for i := range in {
				g, il := i/ig, i%ig
				for ol := range og {
					o := g*og + ol
					k := tw[(i*og+ol)*5 : (i*og+ol)*5+5]
					e, d := c.w[(o*ig+il)*3:], c.wOdd[(o*ig+il)*3:]
					e[0], e[1], e[2] = k[4], k[2], k[0]
					d[1], d[2] = k[3], k[1]
				}
			}
			foldBN(c.wOdd, nil, scale, shift)
		}
	} else {
		c.w = l.get(p+".conv.weight", out, ig, 1, 5)
	}
	foldBN(c.w, c.b, scale, shift)
	if act == actPReLU {
		c.alpha = loadAlpha(l, p+".act.weight")
	}
	return c
}

// outLen is the output width for input width f.
func (c *convF) outLen(f int) int {
	if c.transposed {
		return 2*f - 1
	}
	return (f-1)/2 + 1
}

func (c *convF) forward(dst, src []float32, fin int, s *scratch) {
	ig, og := c.in/c.groups, c.out/c.groups
	fout := c.outLen(fin)
	if c.transposed {
		// Even outputs 2e and odd outputs 2e+1 are separate products over
		// the rows x[e-1], x[e], x[e+1], interleaved afterwards.
		even, odd := s.even[:c.out*fin], s.odd[:c.out*fin]
		for g := range c.groups {
			col := s.col[:ig*3*fin]
			for il := range ig {
				x := src[(g*ig+il)*fin : (g*ig+il+1)*fin]
				t := col[il*3*fin : (il+1)*3*fin]
				t[0] = 0
				copy(t[1:fin], x)
				copy(t[fin:2*fin], x)
				copy(t[2*fin:], x[1:])
				t[3*fin-1] = 0
			}
			gemm(even[g*og*fin:], fin, c.w[g*og*ig*3:], col, fin, c.b[g*og:], og, ig*3, fin)
			gemm(odd[g*og*fin:], fin, c.wOdd[g*og*ig*3:], col, fin, c.b[g*og:], og, ig*3, fin-1)
		}
		for o := range c.out {
			row := dst[o*fout : (o+1)*fout]
			e, d := even[o*fin:(o+1)*fin], odd[o*fin:(o+1)*fin]
			for i := range fin - 1 {
				row[2*i], row[2*i+1] = e[i], d[i]
			}
			row[fout-1] = e[fin-1]
		}
	} else {
		// Output fo reads x[2*fo-2+k]; row k of each input's block holds
		// that for every fo. Splitting x into even and odd samples makes each
		// row a shifted copy.
		for g := range c.groups {
			col := s.col[:ig*5*fout]
			ev, od := s.even[:fout+1], s.odd[:fout+1]
			for il := range ig {
				x := src[(g*ig+il)*fin : (g*ig+il+1)*fin]
				clear(ev)
				clear(od)
				for i, v := range x {
					if i&1 == 0 {
						ev[i/2] = v
					} else {
						od[i/2] = v
					}
				}
				t := col[il*5*fout : (il+1)*5*fout]
				// Taps 0-4 read ev[fo-1], od[fo-1], ev[fo], od[fo], ev[fo+1].
				t[0], t[fout] = 0, 0
				copy(t[1:fout], ev)
				copy(t[fout+1:2*fout], od)
				copy(t[2*fout:3*fout], ev)
				copy(t[3*fout:4*fout], od)
				copy(t[4*fout:5*fout], ev[1:])
			}
			gemm(dst[g*og*fout:], fout, c.w[g*og*ig*5:], col, fout, c.b[g*og:], og, ig*5, fout)
		}
	}
	activate(dst[:c.out*fout], c.act, c.alpha)
}

// pointwise is a 1x1 convolution, BatchNorm and an optional PReLU.
type pointwise struct {
	in, out int
	w, b    []float32 // [out][in]
	act     int
	alpha   float32
}

func loadPointwise(l *loader, p, bn string, in, out int, transposed bool, act int, alphaName string) pointwise {
	c := pointwise{in: in, out: out, act: act}
	if transposed {
		w := l.get(p+".weight", in, out, 1, 1)
		if w != nil {
			c.w = make([]float32, in*out)
			for i := range in {
				for o := range out {
					c.w[o*in+i] = w[i*out+o]
				}
			}
		}
	} else {
		c.w = l.get(p+".weight", out, in, 1, 1)
	}
	c.b = l.get(p+".bias", out)
	scale, shift := loadBN(l, bn, out)
	foldBN(c.w, c.b, scale, shift)
	if act == actPReLU {
		c.alpha = loadAlpha(l, alphaName)
	}
	return c
}

// forward writes rows of stride os.
func (c *pointwise) forward(dst []float32, os int, src []float32, f int) {
	gemm(dst, os, c.w, src, f, c.b, c.out, c.in, f)
	activate(dst[:(c.out-1)*os+f], c.act, c.alpha)
}

// depthwise is the 3x3 per-channel convolution of a GT block: three taps in
// time, dilation apart, and three in frequency with padding 1.
type depthwise struct {
	w, b  []float32 // [16][3 time][3 freq]
	alpha float32
}

// forward reads the frames at t-2d, t-d and t, stored with padded rows of
// width padLen.
func (c *depthwise) forward(dst []float32, frames [3][]float32) {
	dwconv(dst, midLen, frames, padLen, c.w, c.b, gtHidden, midLen)
	vecPReLU(dst[:gtHidden*midLen], c.alpha)
}

// gru is torch.nn.GRU, one layer, one direction, gate order r, z, n.
type gru struct {
	in, hid  int
	wih, whh []float32 // [3*hid][in], [3*hid][hid]
	bih, bhh []float32
}

func loadGRU(l *loader, p, suffix string, in, hid int) gru {
	return gru{in: in, hid: hid,
		wih: l.get(p+".weight_ih_l0"+suffix, 3*hid, in),
		whh: l.get(p+".weight_hh_l0"+suffix, 3*hid, hid),
		bih: l.get(p+".bias_ih_l0"+suffix, 3*hid),
		bhh: l.get(p+".bias_hh_l0"+suffix, 3*hid),
	}
}

// step advances h in place. gi and gh are scratch of length 3*hid.
func (g *gru) step(h, x, gi, gh []float32) {
	matvec(gi, g.wih, g.bih, x)
	matvec(gh, g.whh, g.bhh, h)
	gruGates(h, gi, gh)
}

// gruGates finishes a step, or a batch of independent steps, from the input
// and hidden projections laid out [r z n], each len(h). It overwrites gi.
func gruGates(h, gi, gh []float32) {
	n := len(h)
	rz := gi[:2*n]
	for i, v := range gh[:2*n] {
		rz[i] += v
	}
	vecSigmoid(rz, rz)
	c, r, ghn := gi[2*n:3*n], gi[:n], gh[2*n:3*n]
	for i := range c {
		c[i] += r[i] * ghn[i]
	}
	vecTanh(c, c)
	z := gi[n : 2*n]
	for i := range h {
		h[i] = c[i] + z[i]*(h[i]-c[i])
	}
}

// linear is torch.nn.Linear.
type linear struct{ w, b []float32 }

// matvec sets y = W x + b for W stored [len(y)][len(x)]. Every layer here
// has 4, 8 or 16 inputs.
func matvec(y, w, b, x []float32) {
	b = b[:len(y)]
	switch len(x) {
	case 4:
		v := (*[4]float32)(x)
		for i := range y {
			r := (*[4]float32)(w[4*i:])
			y[i] = b[i] + (r[0]*v[0] + r[1]*v[1]) + (r[2]*v[2] + r[3]*v[3])
		}
	case 8:
		v := (*[8]float32)(x)
		for i := range y {
			r := (*[8]float32)(w[8*i:])
			y[i] = b[i] + ((r[0]*v[0] + r[1]*v[1]) + (r[2]*v[2] + r[3]*v[3])) +
				((r[4]*v[4] + r[5]*v[5]) + (r[6]*v[6] + r[7]*v[7]))
		}
	case 16:
		v := (*[16]float32)(x)
		for i := range y {
			r := (*[16]float32)(w[16*i:])
			y[i] = b[i] + (((r[0]*v[0] + r[1]*v[1]) + (r[2]*v[2] + r[3]*v[3])) +
				((r[4]*v[4] + r[5]*v[5]) + (r[6]*v[6] + r[7]*v[7]))) +
				(((r[8]*v[8] + r[9]*v[9]) + (r[10]*v[10] + r[11]*v[11])) +
					((r[12]*v[12] + r[13]*v[13]) + (r[14]*v[14] + r[15]*v[15])))
		}
	default:
		panic("gtcrn: matvec width")
	}
}

// banded is a matrix whose rows are nonzero only over a contiguous span, as
// the ERB filterbank's are.
type banded struct {
	start []int
	w     [][]float32
}

func newBanded(dense []float32, rows, cols int) banded {
	var b banded
	if dense == nil {
		return b
	}
	for r := range rows {
		row := dense[r*cols : (r+1)*cols]
		lo, hi := 0, cols
		for lo < hi && row[lo] == 0 {
			lo++
		}
		for hi > lo && row[hi-1] == 0 {
			hi--
		}
		b.start = append(b.start, lo)
		b.w = append(b.w, row[lo:hi])
	}
	return b
}

func (b *banded) apply(y, x []float32) {
	for r, w := range b.w {
		xs := x[b.start[r]:]
		var acc float32
		for j, v := range w {
			acc += v * xs[j]
		}
		y[r] = acc
	}
}
