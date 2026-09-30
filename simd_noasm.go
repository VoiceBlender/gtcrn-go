//go:build !amd64

package gtcrn

// useFMA mirrors the amd64 flag so tests build everywhere.
var useFMA = false

func gemmFast(out []float32, os int, w, x []float32, xs int, b []float32, rows, inner, n int) bool {
	return false
}

func vecSigmoidFast(dst, src []float32) int { return 0 }

func vecTanhFast(dst, src []float32) int { return 0 }

func dwconvFast(out []float32, os int, x [3][]float32, xs int, w, b []float32, ch, n int) bool {
	return false
}

func vecPReLUFast(x []float32, alpha float32) int { return 0 }
