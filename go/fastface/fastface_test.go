package fastface

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// exeName returns the OS-specific binary name for fastface_int8.
func exeName(root string) string {
	suffix := ""
	if runtime.GOOS == "windows" {
		suffix = ".exe"
	}
	for _, name := range []string{"fastface_int8" + suffix, "fastface_int8.exe", "fastface_int8"} {
		p := filepath.Join(root, name)
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return filepath.Join(root, "fastface_int8"+suffix)
}

// TestGolden verifies the Go wrapper produces bit-exact embedding matching
// the committed tests/golden_int8_emb.bin reference.
func TestGolden(t *testing.T) {
	root := findRepoRoot(t)
	goldenInputPath := filepath.Join(root, "tests", "golden_input.bin")
	goldenEmbPath := filepath.Join(root, "tests", "golden_int8_emb.bin")
	exe := exeName(root)
	weights := filepath.Join(root, "models", "w600k_r50_ffw4.bin")

	inputBytes, err := os.ReadFile(goldenInputPath)
	if err != nil {
		t.Fatalf("read golden input: %v", err)
	}
	if len(inputBytes) != InputBytes {
		t.Fatalf("golden input wrong size: %d", len(inputBytes))
	}
	input := make([]float32, InputSize)
	for i := range input {
		input[i] = math.Float32frombits(binary.LittleEndian.Uint32(inputBytes[i*4:]))
	}

	ff, err := New(Config{Exe: exe, Weights: weights, Workdir: root})
	if err != nil {
		t.Fatalf("fastface.New: %v", err)
	}
	defer ff.Close()

	emb, err := ff.Embed(input)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	goldenBytes, err := os.ReadFile(goldenEmbPath)
	if err != nil {
		t.Fatalf("read golden emb: %v", err)
	}
	golden := make([]float32, OutputSize)
	for i := range golden {
		golden[i] = math.Float32frombits(binary.LittleEndian.Uint32(goldenBytes[i*4:]))
	}

	mismatches := 0
	for i := range emb {
		if emb[i] != golden[i] {
			mismatches++
			if mismatches <= 3 {
				t.Errorf("emb[%d] = %g, want %g", i, emb[i], golden[i])
			}
		}
	}
	if mismatches == 0 {
		t.Logf("PASS: %d embeddings bit-exact vs golden", OutputSize)
	} else {
		t.Errorf("total mismatches: %d/%d", mismatches, OutputSize)
	}
}

func findRepoRoot(t *testing.T) string {
	abs, _ := filepath.Abs(".")
	for {
		if _, err := os.Stat(filepath.Join(abs, "fastface.h")); err == nil {
			return abs
		}
		parent := filepath.Dir(abs)
		if parent == abs {
			t.Fatalf("could not find repo root from %s", abs)
		}
		abs = parent
	}
}
