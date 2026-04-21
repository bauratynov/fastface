package fastface

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// TestConcurrentInstances runs 2 FastFace subprocesses in parallel, each on
// a dedicated goroutine, verifies bit-exact output vs golden across all
// inferences, and measures throughput vs sequential.
func TestConcurrentInstances(t *testing.T) {
	root := findRepoRoot(t)
	goldenInputPath := filepath.Join(root, "tests", "golden_input.bin")
	goldenEmbPath := filepath.Join(root, "tests", "golden_int8_emb.bin")
	exe := exeName(root)
	weights := filepath.Join(root, "models", "w600k_r50_ffw4.bin")

	inputBytes, err := os.ReadFile(goldenInputPath)
	if err != nil {
		t.Fatalf("read golden input: %v", err)
	}
	input := make([]float32, InputSize)
	for i := range input {
		input[i] = math.Float32frombits(binary.LittleEndian.Uint32(inputBytes[i*4:]))
	}
	goldenBytes, err := os.ReadFile(goldenEmbPath)
	if err != nil {
		t.Fatalf("read golden emb: %v", err)
	}
	golden := make([]float32, OutputSize)
	for i := range golden {
		golden[i] = math.Float32frombits(binary.LittleEndian.Uint32(goldenBytes[i*4:]))
	}

	const NPerGoroutine = 100
	const NGoroutines = 2

	var wg sync.WaitGroup
	errs := make([]error, NGoroutines)
	mismatches := make([]int, NGoroutines)

	// Use --threads 4 per instance so 2 instances don't starve each other
	// on an 8-P-core machine.
	t0 := time.Now()
	for g := 0; g < NGoroutines; g++ {
		wg.Add(1)
		go func(gid int) {
			defer wg.Done()
			ff, err := New(Config{Exe: exe, Weights: weights, Workdir: root})
			if err != nil {
				errs[gid] = err
				return
			}
			defer ff.Close()
			for i := 0; i < NPerGoroutine; i++ {
				emb, err := ff.Embed(input)
				if err != nil {
					errs[gid] = err
					return
				}
				for j := range emb {
					if emb[j] != golden[j] {
						mismatches[gid]++
						break
					}
				}
			}
		}(g)
	}
	wg.Wait()
	elapsed := time.Since(t0)

	totalEmbeds := NGoroutines * NPerGoroutine
	for g := 0; g < NGoroutines; g++ {
		if errs[g] != nil {
			t.Errorf("goroutine %d: %v", g, errs[g])
		}
		if mismatches[g] > 0 {
			t.Errorf("goroutine %d: %d mismatches vs golden", g, mismatches[g])
		}
	}
	if !t.Failed() {
		t.Logf("PASS: %d goroutines x %d embeds = %d total in %.2fs (%.1f face/s aggregate)",
			NGoroutines, NPerGoroutine, totalEmbeds, elapsed.Seconds(),
			float64(totalEmbeds)/elapsed.Seconds())
	}
}
