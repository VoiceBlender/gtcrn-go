package gtcrn

// Network shape, fixed by the trained weights.
const (
	bins     = 257 // 512-point spectrum at 16 kHz
	erbLow   = 65  // bins kept as they are below the ERB bands
	erbBands = 64
	featLen  = erbLow + erbBands // 129
	midLen   = 33                // frequency width after the two strided convs
	gtHidden = 16                // channels through a GT block
	gtHalf   = 8
	padLen   = midLen + 3 // a depthwise input row: one zero left, two right
)

var dilations = [3]int{1, 2, 5}

// gtBlock is StreamGTConvBlock: half the channels pass through, the other half
// go through subband feature extraction, pointwise, dilated depthwise over
// time, pointwise and temporal attention, and the halves are interleaved.
type gtBlock struct {
	dil      int
	pw1, pw2 pointwise
	dw       depthwise
	traGRU   gru // 8 -> 16
	traFC    linear
}

func loadGTBlock(l *loader, p string, dil int, transposed bool) gtBlock {
	dwName := ".depth_conv.Conv2d"
	if transposed {
		dwName = ".depth_conv.ConvTranspose2d"
	}
	b := gtBlock{
		dil: dil,
		pw1: loadPointwise(l, p+".point_conv1", p+".point_bn1", 3*gtHalf, gtHidden, transposed, actPReLU, p+".point_act.weight"),
		pw2: loadPointwise(l, p+".point_conv2", p+".point_bn2", gtHidden, gtHalf, transposed, actNone, ""),
		dw: depthwise{
			w:     l.get(p+dwName+".weight", gtHidden, 1, 3, 3),
			b:     l.get(p+dwName+".bias", gtHidden),
			alpha: loadAlpha(l, p+".depth_act.weight"),
		},
		traGRU: loadGRU(l, p+".tra.att_gru", "", gtHalf, 2*gtHalf),
		traFC:  linear{l.get(p+".tra.att_fc.weight", gtHalf, 2*gtHalf), l.get(p+".tra.att_fc.bias", gtHalf)},
	}
	scale, shift := loadBN(l, p+".depth_bn", gtHidden)
	foldBN(b.dw.w, b.dw.b, scale, shift)
	return b
}

// gtState is one GT block's history: the depthwise conv's last 2*dil input
// frames as a ring, oldest at pos, and the attention GRU's hidden state.
type gtState struct {
	hist [][]float32
	pos  int
	traH [2 * gtHalf]float32
}

// sfe is SFE with kernel 3: each channel becomes three, the bin itself and its
// two neighbours, zero at the edges.
func sfe(dst, src []float32, ch, f int) {
	for c := range ch {
		x := src[c*f : (c+1)*f]
		d0, d1, d2 := dst[(3*c)*f:(3*c+1)*f], dst[(3*c+1)*f:(3*c+2)*f], dst[(3*c+2)*f:(3*c+3)*f]
		d0[0] = 0
		copy(d0[1:], x[:f-1])
		copy(d1, x)
		copy(d2, x[1:])
		d2[f-1] = 0
	}
}

func (b *gtBlock) forward(dst, src []float32, st *gtState, s *scratch) {
	const f = midLen
	x2 := src[gtHalf*f:]
	sfe(s.sfe[:3*gtHalf*f], src[:gtHalf*f], gtHalf, f)
	// The depthwise conv's inputs, this frame and the history, are kept in
	// padded rows so its frequency taps need no edge cases. The pads stay zero.
	cur := s.gtH[:]
	b.pw1.forward(cur[1:], padLen, s.sfe[:3*gtHalf*f], f)

	n := 2 * b.dil
	h2 := s.gtH2[:gtHidden*f]
	b.dw.forward(h2, [3][]float32{st.hist[st.pos], st.hist[(st.pos+b.dil)%n], cur})
	copy(st.hist[st.pos], cur)
	st.pos = (st.pos + 1) % n

	h3 := s.gtH3[:gtHalf*f]
	b.pw2.forward(h3, f, h2, f)

	// Temporal recurrent attention: per-channel energy through a GRU gives a
	// per-channel gain.
	var z, a [gtHalf]float32
	for c := range gtHalf {
		var e float32
		for _, v := range h3[c*f : (c+1)*f] {
			e += v * v
		}
		z[c] = e / f
	}
	b.traGRU.step(st.traH[:], z[:], s.gi[:6*gtHalf], s.gh[:6*gtHalf])
	matvec(a[:], b.traFC.w, b.traFC.b, st.traH[:])
	for c := range gtHalf {
		g := sigmoidf(a[c])
		row := h3[c*f : (c+1)*f]
		for i := range row {
			row[i] *= g
		}
		copy(dst[(2*c)*f:(2*c+1)*f], row)
		copy(dst[(2*c+1)*f:(2*c+2)*f], x2[c*f:(c+1)*f])
	}
}

// layerNorm normalises over the whole frame, all 16x33 values, with a
// per-element affine stored channel-major.
type layerNorm struct{ w, b []float32 }

// loadLN transposes torch's [33][16] affine to [16][33].
func loadLN(l *loader, p string) layerNorm {
	n := layerNorm{make([]float32, midLen*gtHidden), make([]float32, midLen*gtHidden)}
	w, b := l.get(p+".weight", midLen, gtHidden), l.get(p+".bias", midLen, gtHidden)
	if l.err != nil {
		return n
	}
	for j := range midLen {
		for c := range gtHidden {
			n.w[c*midLen+j], n.b[c*midLen+j] = w[j*gtHidden+c], b[j*gtHidden+c]
		}
	}
	return n
}

func (n *layerNorm) apply(x []float32) {
	// Four partial sums each: 528 is a multiple of four.
	var s0, s1, s2, s3 float64
	for i := 0; i+4 <= len(x); i += 4 {
		s0 += float64(x[i])
		s1 += float64(x[i+1])
		s2 += float64(x[i+2])
		s3 += float64(x[i+3])
	}
	mean := (s0 + s1 + s2 + s3) / float64(len(x))
	s0, s1, s2, s3 = 0, 0, 0, 0
	for i := 0; i+4 <= len(x); i += 4 {
		d0, d1 := float64(x[i])-mean, float64(x[i+1])-mean
		d2, d3 := float64(x[i+2])-mean, float64(x[i+3])-mean
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	inv := 1 / sqrt64((s0+s1+s2+s3)/float64(len(x))+1e-8)
	m, s := float32(mean), float32(inv)
	w, b := n.w[:len(x)], n.b[:len(x)]
	for i, v := range x {
		x[i] = (v-m)*s*w[i] + b[i]
	}
}

// dpgrnn is DPGRNN: a bidirectional GRU across frequency within the frame,
// then a unidirectional GRU across time per frequency. Each GRU is split into
// two groups over the channels.
type dpgrnn struct {
	intra            [2][2]gru // [group][forward, backward], 8 -> 4
	intraFC, interFC linear
	intraLN, interLN layerNorm
	inter            [2]gru // 8 -> 8
}

func loadDPGRNN(l *loader, p string) dpgrnn {
	var d dpgrnn
	for g := range 2 {
		rnn := fmtRNN(p+".intra_rnn", g)
		d.intra[g][0] = loadGRU(l, rnn, "", gtHalf, 4)
		d.intra[g][1] = loadGRU(l, rnn, "_reverse", gtHalf, 4)
		d.inter[g] = loadGRU(l, fmtRNN(p+".inter_rnn", g), "", gtHalf, gtHalf)
	}
	d.intraFC = linear{l.get(p+".intra_fc.weight", gtHidden, gtHidden), l.get(p+".intra_fc.bias", gtHidden)}
	d.interFC = linear{l.get(p+".inter_fc.weight", gtHidden, gtHidden), l.get(p+".inter_fc.bias", gtHidden)}
	d.intraLN = loadLN(l, p+".intra_ln")
	d.interLN = loadLN(l, p+".inter_ln")
	return d
}

func fmtRNN(p string, g int) string {
	return p + ".rnn" + string(rune('1'+g))
}

// forward runs in place on x, [16][33]. h is the inter GRU state, also
// [16][33].
func (d *dpgrnn) forward(x, h []float32, s *scratch) {
	const f, c = midLen, gtHidden

	// Intra: four recurrences across frequency, two groups by two
	// directions, all independent. Project every input first, then step the
	// four in lockstep so their gates share vector activations.
	for ch := range 4 {
		rnn := &d.intra[ch/2][ch%2]
		gi := s.intraGI[ch][:]
		gemm(gi, f, rnn.wih, x[ch/2*gtHalf*f:], f, rnn.bih, 12, gtHalf, f)
	}
	y := s.y[:c*f]
	var hs [16]float32 // [chain][4]
	var rz [32]float32 // [chain][r0-3 z0-3]
	var nn [16]float32 // [chain][4]
	var gh [48]float32 // [chain][r z n]x4
	var col [4]int     // each chain's frequency this step
	for k := range f {
		for ch := range 4 {
			col[ch] = k
			if ch%2 == 1 {
				col[ch] = f - 1 - k
			}
			rnn := &d.intra[ch/2][ch%2]
			g := gh[ch*12 : ch*12+12]
			matvec(g, rnn.whh, rnn.bhh, hs[ch*4:ch*4+4])
			gi, j := &s.intraGI[ch], col[ch]
			for u := range 8 {
				rz[ch*8+u] = gi[u*f+j] + g[u]
			}
		}
		vecSigmoid(rz[:], rz[:])
		for ch := range 4 {
			gi, j := &s.intraGI[ch], col[ch]
			for u := range 4 {
				nn[ch*4+u] = gi[(8+u)*f+j] + rz[ch*8+u]*gh[ch*12+8+u]
			}
		}
		vecTanh(nn[:], nn[:])
		for ch := range 4 {
			out := (ch/2*gtHalf + ch%2*4) * f
			for u := range 4 {
				hv, n := hs[ch*4+u], nn[ch*4+u]
				hv = n + rz[ch*8+4+u]*(hv-n)
				hs[ch*4+u] = hv
				y[out+u*f+col[ch]] = hv
			}
		}
	}
	io := s.io[:c*f]
	gemm(io, f, d.intraFC.w, y, f, d.intraFC.b, c, c, f)
	d.intraLN.apply(io)
	for i := range io {
		io[i] += x[i]
	}

	// Inter: one step in time for all 33 frequencies at once.
	for g := range 2 {
		rnn := &d.inter[g]
		gi, gh := s.interGI[:], s.interGH[:]
		gemm(gi, f, rnn.wih, io[g*gtHalf*f:], f, rnn.bih, 3*gtHalf, gtHalf, f)
		gemm(gh, f, rnn.whh, h[g*gtHalf*f:], f, rnn.bhh, 3*gtHalf, gtHalf, f)
		gruGates(h[g*gtHalf*f:(g+1)*gtHalf*f], gi, gh)
	}
	z := s.z[:c*f]
	gemm(z, f, d.interFC.w, h, f, d.interFC.b, c, c, f)
	d.interLN.apply(z)
	for i := range x {
		x[i] = io[i] + z[i]
	}
}

// state is everything a stream carries between frames.
type state struct {
	en, de [3]gtState
	inter  [2][]float32 // [16][33]
}

func newState() *state {
	st := &state{}
	for i, d := range dilations {
		for _, g := range []*gtState{&st.en[i], &st.de[2-i]} {
			g.hist = make([][]float32, 2*d)
			for k := range g.hist {
				g.hist[k] = make([]float32, gtHidden*padLen)
			}
		}
	}
	for i := range st.inter {
		st.inter[i] = make([]float32, midLen*gtHidden)
	}
	return st
}

func (st *state) reset() {
	for _, gs := range [][3]gtState{st.en, st.de} {
		for i := range gs {
			for _, h := range gs[i].hist {
				clear(h)
			}
		}
	}
	for i := range 3 {
		st.en[i].pos, st.de[i].pos = 0, 0
		st.en[i].traH, st.de[i].traH = [16]float32{}, [16]float32{}
	}
	for _, h := range st.inter {
		clear(h)
	}
}

type scratch struct {
	feat    [3 * bins]float32
	erb     [3 * featLen]float32
	sfeIn   [9 * featLen]float32
	en      [5][gtHidden * 65]float32 // encoder outputs, kept for the skips
	a, b    [gtHidden * 65]float32
	mask    [2 * featLen]float32
	col     [16 * 3 * 65]float32 // convF's shifted input rows
	even    [gtHidden * 65]float32
	odd     [gtHidden * 65]float32
	sfe     [3 * gtHalf * midLen]float32
	gtH     [gtHidden * padLen]float32
	gtH2    [gtHidden * midLen]float32
	gtH3    [gtHalf * midLen]float32
	y, z    [gtHidden * midLen]float32
	io      [gtHidden * midLen]float32
	gi, gh  [48]float32
	intraGI [4][12 * midLen]float32
	interGI [3 * gtHalf * midLen]float32
	interGH [3 * gtHalf * midLen]float32
	pcm     [256]float32
}

// forward computes the complex mask for one frame. re and im are the 257-bin
// spectrum at the 512-point level; mask receives 2x257, real then imaginary.
func (m *Model) forward(mask, re, im []float32, st *state, s *scratch) {
	feat := s.feat[:]
	for i := range bins {
		feat[i] = sqrt32(re[i]*re[i] + im[i]*im[i] + 1e-12)
	}
	copy(feat[bins:], re[:bins])
	copy(feat[2*bins:], im[:bins])
	for c := range 3 {
		src := feat[c*bins : (c+1)*bins]
		dst := s.erb[c*featLen : (c+1)*featLen]
		copy(dst, src[:erbLow])
		m.erb.apply(dst[erbLow:], src[erbLow:])
	}
	sfe(s.sfeIn[:], s.erb[:], 3, featLen)

	e := &s.en
	m.en0.forward(e[0][:16*65], s.sfeIn[:], featLen, s)
	m.en1.forward(e[1][:16*midLen], e[0][:16*65], 65, s)
	m.enGT[0].forward(e[2][:16*midLen], e[1][:16*midLen], &st.en[0], s)
	m.enGT[1].forward(e[3][:16*midLen], e[2][:16*midLen], &st.en[1], s)
	m.enGT[2].forward(e[4][:16*midLen], e[3][:16*midLen], &st.en[2], s)

	x := s.a[:16*midLen]
	copy(x, e[4][:16*midLen])
	m.dp[0].forward(x, st.inter[0], s)
	m.dp[1].forward(x, st.inter[1], s)

	t := s.b[:16*midLen]
	for i, sk := range []int{4, 3, 2} {
		for j := range x {
			t[j] = x[j] + e[sk][j]
		}
		m.deGT[i].forward(x, t, &st.de[i], s)
	}
	for j := range x {
		t[j] = x[j] + e[1][j]
	}
	d3 := s.a[:16*65]
	m.de3.forward(d3, t, midLen, s)
	t = s.b[:16*65]
	for j := range t {
		t[j] = d3[j] + e[0][j]
	}
	m.de4.forward(s.mask[:], t, 65, s)

	for c := range 2 {
		src := s.mask[c*featLen : (c+1)*featLen]
		dst := mask[c*bins : (c+1)*bins]
		copy(dst, src[:erbLow])
		m.ierb.apply(dst[erbLow:], src[erbLow:])
	}
}
