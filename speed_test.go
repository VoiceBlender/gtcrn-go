package gtcrn

import (
	"fmt"
	"math/rand"
	"testing"
)

// BenchmarkProcess reports cost per 10 ms of audio, the unit rnnoise-go's
// BENCHMARKS.md uses, and must stay at zero allocations.
func BenchmarkProcess(b *testing.B) {
	for _, rate := range []int{8000, 12000, 16000} {
		b.Run(fmt.Sprint(rate), func(b *testing.B) {
			d, err := New(Options{SampleRate: rate})
			if err != nil {
				b.Fatal(err)
			}
			in, out := noise(d.FrameSize()), make([]float32, d.FrameSize())
			b.ReportAllocs()
			for b.Loop() {
				if err := d.Process(out, in); err != nil {
					b.Fatal(err)
				}
			}
			per10ms := float64(b.Elapsed().Nanoseconds()) / float64(b.N) / 1.6
			b.ReportMetric(per10ms/1e3, "µs/10ms")
			b.ReportMetric(1e7/per10ms, "xRT")
		})
	}
}

// BenchmarkProcessGeneric is BenchmarkProcess without the assembly, the cost
// on architectures that have none.
func BenchmarkProcessGeneric(b *testing.B) {
	saved := useFMA
	useFMA = false
	defer func() { useFMA = saved }()
	BenchmarkProcess(b)
}

// BenchmarkParallel runs one stream per goroutine on a shared Model. 10 ms
// divided by the reported µs/10ms is sessions per CPU.
func BenchmarkParallel(b *testing.B) {
	for _, rate := range []int{8000, 16000} {
		b.Run(fmt.Sprint(rate), func(b *testing.B) {
			b.RunParallel(func(pb *testing.PB) {
				d, _ := New(Options{SampleRate: rate})
				in, out := noise(d.FrameSize()), make([]float32, d.FrameSize())
				for pb.Next() {
					d.Process(out, in)
				}
			})
			per10ms := float64(b.Elapsed().Nanoseconds()) / float64(b.N) / 1.6
			b.ReportMetric(per10ms/1e3, "µs/10ms")
			b.ReportMetric(1e7/per10ms, "sessions")
		})
	}
}

func noise(n int) []float32 {
	rng := rand.New(rand.NewSource(1))
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(0.1 * rng.NormFloat64())
	}
	return x
}
