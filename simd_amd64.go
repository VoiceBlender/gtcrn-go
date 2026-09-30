//go:build amd64

package gtcrn

import "golang.org/x/sys/cpu"

// useFMA gates every kernel in simd_amd64.s. Tests turn it off to run the
// generic path.
var useFMA = cpu.X86.HasAVX2 && cpu.X86.HasFMA

// gemmAVX2 takes any n > 0; the activation kernels take n a multiple of 8.
//
//go:noescape
func gemmAVX2(out *float32, os int, w, x *float32, xs int, b *float32, rows, inner, n int)

//go:noescape
func dwconvAVX2(out *float32, os int, x0, x1, x2 *float32, xs int, w, b *float32, ch, n int)

//go:noescape
func vecPReLUAVX2(x *float32, n int, alpha float32)

//go:noescape
func vecSigmoidAVX2(dst, src *float32, n int)

//go:noescape
func vecTanhAVX2(dst, src *float32, n int)

func gemmFast(out []float32, os int, w, x []float32, xs int, b []float32, rows, inner, n int) bool {
	if !useFMA {
		return false
	}
	gemmAVX2(&out[0], os, &w[0], &x[0], xs, &b[0], rows, inner, n)
	return true
}

func dwconvFast(out []float32, os int, x [3][]float32, xs int, w, b []float32, ch, n int) bool {
	if !useFMA {
		return false
	}
	dwconvAVX2(&out[0], os, &x[0][0], &x[1][0], &x[2][0], xs, &w[0], &b[0], ch, n)
	return true
}

func vecPReLUFast(x []float32, alpha float32) int {
	n8 := len(x) &^ 7
	if !useFMA || n8 == 0 {
		return 0
	}
	vecPReLUAVX2(&x[0], n8, alpha)
	return n8
}

func vecSigmoidFast(dst, src []float32) int {
	n8 := len(src) &^ 7
	if !useFMA || n8 == 0 {
		return 0
	}
	vecSigmoidAVX2(&dst[0], &src[0], n8)
	return n8
}

func vecTanhFast(dst, src []float32) int {
	n8 := len(src) &^ 7
	if !useFMA || n8 == 0 {
		return 0
	}
	vecTanhAVX2(&dst[0], &src[0], n8)
	return n8
}
