package gtcrn

// Kernels with an assembly fast path. Each wrapper checks bounds and hands
// over to the architecture's *Fast when it can. The generic code doubles as
// the test oracle.

// gemm sets the first n columns of rows rows of out to b + W·X:
// out[r*os+j] = b[r] + Σ_i w[r*inner+i] · x[i*xs+j].
func gemm(out []float32, os int, w, x []float32, xs int, b []float32, rows, inner, n int) {
	if rows == 0 || inner == 0 || n == 0 {
		return
	}
	_ = out[(rows-1)*os+n-1]
	_ = x[(inner-1)*xs+n-1]
	_ = w[rows*inner-1]
	_ = b[rows-1]
	if !gemmFast(out, os, w, x, xs, b, rows, inner, n) {
		gemmGeneric(out, os, w, x, xs, b, rows, inner, n)
	}
}

func gemmGeneric(out []float32, os int, w, x []float32, xs int, b []float32, rows, inner, n int) {
	for r := range rows {
		row := out[r*os : r*os+n]
		for j := range row {
			row[j] = b[r]
		}
		wr := w[r*inner : (r+1)*inner]
		i := 0
		// Four input rows per pass over the output row.
		for ; i+4 <= inner; i += 4 {
			w0, w1, w2, w3 := wr[i], wr[i+1], wr[i+2], wr[i+3]
			x0 := x[i*xs : i*xs+n][:len(row)]
			x1 := x[(i+1)*xs : (i+1)*xs+n][:len(row)]
			x2 := x[(i+2)*xs : (i+2)*xs+n][:len(row)]
			x3 := x[(i+3)*xs : (i+3)*xs+n][:len(row)]
			for j := range row {
				row[j] += w0*x0[j] + w1*x1[j] + w2*x2[j] + w3*x3[j]
			}
		}
		for ; i < inner; i++ {
			wi, xr := wr[i], x[i*xs : i*xs+n][:len(row)]
			for j := range row {
				row[j] += wi * xr[j]
			}
		}
	}
}

// dwconv is the GT block's depthwise 3x3. x holds the frames at t-2d, t-d
// and t as rows of stride xs, each zero-padded by one on the left and at least
// one on the right: out[c*os+j] = b[c] + Σ_{t,k} w[c*9+t*3+k] · x[t][c*xs+j+k].
func dwconv(out []float32, os int, x [3][]float32, xs int, w, b []float32, ch, n int) {
	_ = out[(ch-1)*os+n-1]
	for _, r := range x {
		_ = r[(ch-1)*xs+n+1]
	}
	_ = w[ch*9-1]
	_ = b[ch-1]
	if dwconvFast(out, os, x, xs, w, b, ch, n) {
		return
	}
	for c := range ch {
		row := out[c*os : c*os+n]
		for j := range row {
			row[j] = b[c]
		}
		for t, fr := range x {
			xr := fr[c*xs : c*xs+n+2]
			k := w[c*9+t*3 : c*9+t*3+3]
			for j := range row {
				row[j] += k[0]*xr[j] + k[1]*xr[j+1] + k[2]*xr[j+2]
			}
		}
	}
}

func vecPReLU(x []float32, alpha float32) {
	for i := vecPReLUFast(x, alpha); i < len(x); i++ {
		if x[i] < 0 {
			x[i] *= alpha
		}
	}
}

// vecSigmoid and vecTanh work in place or between equal-length slices.
func vecSigmoid(dst, src []float32) {
	_ = dst[len(src)-1]
	for i := vecSigmoidFast(dst, src); i < len(src); i++ {
		dst[i] = sigmoidf(src[i])
	}
}

func vecTanh(dst, src []float32) {
	_ = dst[len(src)-1]
	for i := vecTanhFast(dst, src); i < len(src); i++ {
		dst[i] = tanhf(src[i])
	}
}
