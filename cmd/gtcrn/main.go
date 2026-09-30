// Command gtcrn denoises a 16-bit PCM WAV file at 8, 12 or 16 kHz.
//
// The 16 ms delay is compensated, so the output is the same length as the
// input and aligned with it. Each channel is denoised independently.
//
// Usage:
//
//	gtcrn [flags] input.wav output.wav
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	gtcrn "github.com/VoiceBlender/gtcrn-go"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("gtcrn: ")

	modelPath := flag.String("model", "", "read weights from this file instead of the embedded ones")
	quiet := flag.Bool("q", false, "suppress the summary")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: gtcrn [flags] input.wav output.wav\n\n"+
			"Denoises 16-bit PCM WAV at 8, 12 or 16 kHz.\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() != 2 {
		flag.Usage()
		os.Exit(2)
	}
	if err := run(flag.Arg(0), flag.Arg(1), *modelPath, *quiet); err != nil {
		log.Fatal(strings.TrimPrefix(err.Error(), "gtcrn: "))
	}
}

func run(inPath, outPath, modelPath string, quiet bool) error {
	w, err := readWAV(inPath)
	if err != nil {
		return err
	}
	if w.frames() == 0 {
		return fmt.Errorf("%s: no audio samples", inPath)
	}
	o := gtcrn.Options{SampleRate: w.rate}
	if modelPath != "" {
		if o.Model, err = gtcrn.LoadModelFile(modelPath); err != nil {
			return err
		}
	}

	start := time.Now()
	if err := denoise(w, o); err != nil {
		return err
	}
	elapsed := time.Since(start)
	if err := writeWAV(outPath, w); err != nil {
		return err
	}
	if !quiet {
		audio := time.Duration(float64(w.frames()) / float64(w.rate) * float64(time.Second))
		fmt.Printf("%s: %d Hz, %d channel(s), %s\n", inPath, w.rate, w.channels, audio.Round(time.Millisecond))
		fmt.Printf("denoised in %s (%.0f× real time)\n", elapsed.Round(time.Millisecond), audio.Seconds()/elapsed.Seconds())
		fmt.Printf("wrote %s\n", outPath)
	}
	return nil
}

// denoise replaces w.samples with the denoised signal. Each channel runs
// Delay() samples past the end and drops the first Delay() output samples.
func denoise(w *wavFile, o gtcrn.Options) error {
	d, err := gtcrn.New(o)
	if err != nil {
		return err
	}
	frame, delay := d.FrameSize(), d.Delay()
	n := w.frames()
	nframes := (n + delay + frame - 1) / frame
	out := make([]int16, len(w.samples))
	inBuf, outBuf := make([]int16, frame), make([]int16, frame)
	for ch := range w.channels {
		d.Reset()
		for f := range nframes {
			base := f * frame
			for i := range inBuf {
				inBuf[i] = 0
				if src := base + i; src < n {
					inBuf[i] = w.samples[src*w.channels+ch]
				}
			}
			if err := d.ProcessInt16(outBuf, inBuf); err != nil {
				return err
			}
			for i, v := range outBuf {
				if dst := base + i - delay; dst >= 0 && dst < n {
					out[dst*w.channels+ch] = v
				}
			}
		}
	}
	w.samples = out
	return nil
}
