package gtcrn

import "math"

// expf is exp for float32, to within about 2 ulp over the range the network
// reaches. math.Exp costs several times more and the GRUs call this thousands
// of times per frame.
func expf(x float32) float32 {
	x = min(max(x, -87), 88)
	// Adding 1.5*2^23 rounds to the nearest integer and leaves it in the low
	// mantissa bits.
	t := x*1.44269504 + 12582912
	k := int32(math.Float32bits(t)) - 0x4B400000
	kf := t - 12582912
	r := x - kf*0.693145752 - kf*1.42860677e-6
	p := 1 + r*(1+r*(0.5+r*(0.166666672+r*(0.0416666679+r*(0.00833333377+r*0.00138888892)))))
	return p * math.Float32frombits(uint32(k+127)<<23)
}

func sigmoidf(x float32) float32 { return 1 / (1 + expf(-x)) }

func tanhf(x float32) float32 { return 2/(1+expf(-2*x)) - 1 }

func sqrt32(x float32) float32 { return float32(math.Sqrt(float64(x))) }

func sqrt64(x float64) float64 { return math.Sqrt(x) }
