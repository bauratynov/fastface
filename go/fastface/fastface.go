// Package fastface wraps the FastFace INT8 engine via its --server subprocess.
// Stdlib-only (no cgo). Matches the Python SDK semantics.
//
// Usage (Exe is auto-detected -- .exe on Windows, bare name on Linux/macOS):
//
//	ff, err := fastface.New(fastface.Config{
//	    Weights: `models/w600k_r50_ffw4.bin`,
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer ff.Close()
//
//	// input: HWC float32 [-1, 1], exactly 3*112*112 = 37632 floats.
//	emb, err := ff.Embed(input)  // []float32 length 512
//
// Thread-safety: FastFace is NOT goroutine-safe. Create one instance per
// goroutine that embeds, or wrap with your own sync.Mutex.
package fastface

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
)

// Input face layout constants.
const (
	InputSize   = 3 * 112 * 112 // 37632 floats
	InputBytes  = InputSize * 4
	OutputSize  = 512
	OutputBytes = OutputSize * 4
)

// Config for starting the subprocess.
type Config struct {
	Exe     string // path to fastface_int8[.exe]; auto-detected if empty
	Weights string // path to w600k_r50_ffw4.bin (default "models/w600k_r50_ffw4.bin")
	GCCBin  string // optional dir with libgomp-1.dll for Windows; auto-detected if empty
	Workdir string // cwd for subprocess; default = dir of Exe
}

// FastFace is a persistent subprocess wrapping fastface_int8 --server.
type FastFace struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	stderr io.ReadCloser
	mu     sync.Mutex
	closed bool
}

// defaultExe returns the right binary name for the current OS.
// Windows: fastface_int8.exe; Linux/macOS: fastface_int8 (with .exe fallback).
func defaultExe() string {
	suffix := ""
	if runtime.GOOS == "windows" {
		suffix = ".exe"
	}
	for _, p := range []string{"./fastface_int8" + suffix, "./fastface_int8.exe", "./fastface_int8"} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return "./fastface_int8" + suffix
}

// New starts a subprocess and returns a FastFace ready to Embed.
func New(cfg Config) (*FastFace, error) {
	if cfg.Exe == "" {
		cfg.Exe = defaultExe()
	}
	if cfg.Weights == "" {
		cfg.Weights = "models/w600k_r50_ffw4.bin"
	}
	if cfg.GCCBin == "" && runtime.GOOS == "windows" {
		for _, p := range []string{`C:\mingw64\bin`, `C:/mingw64/bin`} {
			if _, err := os.Stat(p); err == nil {
				cfg.GCCBin = p
				break
			}
		}
	}
	if cfg.Workdir == "" {
		abs, _ := filepath.Abs(cfg.Exe)
		cfg.Workdir = filepath.Dir(abs)
	}

	cmd := exec.Command(cfg.Exe, cfg.Weights, "--server")
	cmd.Dir = cfg.Workdir
	cmd.Env = os.Environ()
	if cfg.GCCBin != "" {
		for i, kv := range cmd.Env {
			if len(kv) >= 5 && (kv[:5] == "PATH=" || kv[:5] == "path=" || kv[:5] == "Path=") {
				cmd.Env[i] = "PATH=" + cfg.GCCBin + string(os.PathListSeparator) + kv[5:]
				break
			}
		}
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	ff := &FastFace{cmd: cmd, stdin: stdin, stdout: stdout, stderr: stderr}
	return ff, nil
}

// Embed runs a single forward pass. input must be exactly 3*112*112 floats
// in HWC order, range roughly [-1, 1]. Returns a new 512-float embedding.
func (ff *FastFace) Embed(input []float32) ([]float32, error) {
	if len(input) != InputSize {
		return nil, fmt.Errorf("input length %d != expected %d", len(input), InputSize)
	}
	ff.mu.Lock()
	defer ff.mu.Unlock()
	if ff.closed {
		return nil, fmt.Errorf("fastface: subprocess already closed")
	}
	// Write fp32 input
	buf := make([]byte, InputBytes)
	for i, v := range input {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	if _, err := ff.stdin.Write(buf); err != nil {
		return nil, fmt.Errorf("stdin write: %w", err)
	}
	// Read fp32 embedding
	outBuf := make([]byte, OutputBytes)
	if _, err := io.ReadFull(ff.stdout, outBuf); err != nil {
		return nil, fmt.Errorf("stdout read: %w", err)
	}
	out := make([]float32, OutputSize)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(outBuf[i*4:]))
	}
	return out, nil
}

// CosSim returns the cosine similarity of two embeddings.
func CosSim(a, b []float32) float32 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}

// Close terminates the subprocess and releases resources.
func (ff *FastFace) Close() error {
	ff.mu.Lock()
	defer ff.mu.Unlock()
	if ff.closed {
		return nil
	}
	ff.closed = true
	_ = ff.stdin.Close()
	err := ff.cmd.Wait()
	return err
}
