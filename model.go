package gtcrn

import (
	"bytes"
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"sync"
)

//go:embed weights.bin
var embedded []byte

// Model holds the network weights, rearranged for the forward pass. It is
// read-only once loaded, so one Model serves any number of Denoisers.
type Model struct {
	erb, ierb banded // 192 bins -> 64 ERB bands, and back

	en0, en1 convF
	enGT     [3]gtBlock
	dp       [2]dpgrnn
	deGT     [3]gtBlock
	de3, de4 convF
}

var (
	defaultOnce  sync.Once
	defaultModel *Model
	defaultErr   error
)

// DefaultModel returns the embedded weights, trained by upstream on the DNS3
// corpus.
func DefaultModel() (*Model, error) {
	defaultOnce.Do(func() { defaultModel, defaultErr = LoadModel(embedded) })
	return defaultModel, defaultErr
}

// LoadModelFile reads a blob written by tools/export.
func LoadModelFile(path string) (*Model, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return LoadModel(b)
}

type tensor struct {
	shape []int
	data  []float32
}

// loader collects the first error so the layer constructors can read
// straight through without checking each tensor.
type loader struct {
	t    map[string]tensor
	used map[string]bool
	err  error
}

func (l *loader) get(name string, shape ...int) []float32 {
	if l.err != nil {
		return nil
	}
	t, ok := l.t[name]
	if !ok {
		l.err = fmt.Errorf("gtcrn: model has no tensor %s", name)
		return nil
	}
	if fmt.Sprint(t.shape) != fmt.Sprint(shape) {
		l.err = fmt.Errorf("gtcrn: tensor %s has shape %v, want %v", name, t.shape, shape)
		return nil
	}
	l.used[name] = true
	return t.data
}

// LoadModel parses a weight blob.
func LoadModel(b []byte) (*Model, error) {
	if !bytes.HasPrefix(b, []byte("GTCRNw01")) || len(b) < 12 {
		return nil, errors.New("gtcrn: not a GTCRNw01 weight blob")
	}
	l := &loader{t: map[string]tensor{}, used: map[string]bool{}}
	n := binary.LittleEndian.Uint32(b[8:])
	b = b[12:]
	short := errors.New("gtcrn: weight blob truncated")
	for range n {
		if len(b) < 2 {
			return nil, short
		}
		nl := int(binary.LittleEndian.Uint16(b))
		if len(b) < 3+nl {
			return nil, short
		}
		name := string(b[2 : 2+nl])
		rank := int(b[2+nl])
		b = b[3+nl:]
		if len(b) < 4*rank {
			return nil, short
		}
		shape := make([]int, rank)
		size := 1
		for i := range shape {
			shape[i] = int(binary.LittleEndian.Uint32(b[4*i:]))
			size *= shape[i]
		}
		b = b[4*rank:]
		if len(b) < 4*size {
			return nil, short
		}
		data := make([]float32, size)
		for i := range data {
			data[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		b = b[4*size:]
		l.t[name] = tensor{shape, data}
	}
	if len(b) != 0 {
		return nil, errors.New("gtcrn: trailing bytes after the last tensor")
	}

	m := &Model{
		erb:  newBanded(l.get("erb.erb_fc.weight", erbBands, bins-erbLow), erbBands, bins-erbLow),
		ierb: newBanded(l.get("erb.ierb_fc.weight", bins-erbLow, erbBands), bins-erbLow, erbBands),
		en0:  loadConvF(l, "encoder.en_convs.0", 9, 16, 1, false, actPReLU),
		en1:  loadConvF(l, "encoder.en_convs.1", 16, 16, 2, false, actPReLU),
		de3:  loadConvF(l, "decoder.de_convs.3", 16, 16, 2, true, actPReLU),
		de4:  loadConvF(l, "decoder.de_convs.4", 16, 2, 1, true, actTanh),
	}
	for i, d := range dilations {
		m.enGT[i] = loadGTBlock(l, fmt.Sprintf("encoder.en_convs.%d", i+2), d, false)
		m.deGT[2-i] = loadGTBlock(l, fmt.Sprintf("decoder.de_convs.%d", 2-i), d, true)
	}
	for i := range m.dp {
		m.dp[i] = loadDPGRNN(l, fmt.Sprintf("dpgrnn%d", i+1))
	}
	if l.err != nil {
		return nil, l.err
	}
	for name := range l.t {
		if !l.used[name] {
			return nil, fmt.Errorf("gtcrn: model has unexpected tensor %s", name)
		}
	}
	return m, nil
}
