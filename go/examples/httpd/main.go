// httpd — minimal REST server wrapping the FastFace SDK.
//
// Exposes:
//   POST /embed    multipart "image" field, image/jpeg or image/png body
//     returns 200 application/json: {"embedding": [512 floats]}
//
//   POST /match    multipart "a" and "b" image fields
//     returns {"cos_sim": 0.7xxx, "same_verdict": "SAME"|"DIFFERENT"}
//
//   GET /health    returns 200 "ok" if subprocess alive
//
// Run:
//   go run main.go --listen :8080
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	fastface "github.com/bauratynov/fastface/go/fastface"
)

const FaceSize = 112
const SameThreshold = 0.20

type server struct {
	ff *fastface.FastFace
	mu sync.Mutex // fastface is not goroutine-safe
}

// preprocessImage center-crops (150 px, shifted up 10) then bilinearly
// resamples to 112x112. Stdlib only.
func preprocessImage(img image.Image) []float32 {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	s := 150
	if w < s {
		s = w
	}
	if h < s {
		s = h
	}
	left := (w - s) / 2
	top := (h-s)/2 - 10
	if top < 0 {
		top = 0
	}
	if top+s > h {
		top = h - s
	}
	arr := make([]float32, FaceSize*FaceSize*3)
	i := 0
	for y := 0; y < FaceSize; y++ {
		fy := float32(y) * float32(s) / float32(FaceSize)
		iy := int(fy)
		dy := fy - float32(iy)
		if iy >= s-1 {
			iy = s - 2
			dy = 1.0
		}
		for x := 0; x < FaceSize; x++ {
			fx := float32(x) * float32(s) / float32(FaceSize)
			ix := int(fx)
			dx := fx - float32(ix)
			if ix >= s-1 {
				ix = s - 2
				dx = 1.0
			}
			var rs, gs, bs float32
			for _, pair := range [4]struct {
				x, y int
				w    float32
			}{
				{ix, iy, (1 - dx) * (1 - dy)},
				{ix + 1, iy, dx * (1 - dy)},
				{ix, iy + 1, (1 - dx) * dy},
				{ix + 1, iy + 1, dx * dy},
			} {
				r, g, b, _ := img.At(bounds.Min.X+left+pair.x, bounds.Min.Y+top+pair.y).RGBA()
				rs += float32(r>>8) * pair.w
				gs += float32(g>>8) * pair.w
				bs += float32(b>>8) * pair.w
			}
			arr[i+0] = (rs - 127.5) / 127.5
			arr[i+1] = (gs - 127.5) / 127.5
			arr[i+2] = (bs - 127.5) / 127.5
			i += 3
		}
	}
	return arr
}

func (s *server) embedImage(img image.Image) ([]float32, error) {
	arr := preprocessImage(img)
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.ff.Embed(arr)
}

func (s *server) handleEmbed(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var img image.Image
	if ct := r.Header.Get("Content-Type"); ct == "image/jpeg" || ct == "image/png" {
		var err error
		img, _, err = image.Decode(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
	} else {
		if err := r.ParseMultipartForm(16 << 20); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		f, _, err := r.FormFile("image")
		if err != nil {
			http.Error(w, "missing 'image' form field", http.StatusBadRequest)
			return
		}
		defer f.Close()
		img, _, err = image.Decode(f)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
	}
	emb, err := s.embedImage(img)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{"embedding": emb})
}

func (s *server) handleMatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	decode := func(field string) (image.Image, error) {
		f, _, err := r.FormFile(field)
		if err != nil {
			return nil, fmt.Errorf("missing %q: %w", field, err)
		}
		defer f.Close()
		img, _, err := image.Decode(f)
		return img, err
	}
	imgA, err := decode("a")
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	imgB, err := decode("b")
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	eA, err := s.embedImage(imgA)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	eB, err := s.embedImage(imgB)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	sim := fastface.CosSim(eA, eB)
	verdict := "DIFFERENT"
	if sim >= SameThreshold {
		verdict = "SAME"
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"cos_sim":      sim,
		"same_verdict": verdict,
		"threshold":    SameThreshold,
	})
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

func findRepoRoot(from string) string {
	abs, _ := filepath.Abs(from)
	for {
		if _, err := os.Stat(filepath.Join(abs, "fastface.h")); err == nil {
			return abs
		}
		parent := filepath.Dir(abs)
		if parent == abs {
			return ""
		}
		abs = parent
	}
}

func main() {
	listen := flag.String("listen", ":8080", "HTTP listen addr")
	flag.Parse()

	root := findRepoRoot(".")
	if root == "" {
		log.Fatal("could not find FastFace repo root (looking for fastface.h)")
	}
	ff, err := fastface.New(fastface.Config{
		Exe:     filepath.Join(root, "fastface_int8.exe"),
		Weights: filepath.Join(root, "models", "w600k_r50_ffw4.bin"),
		Workdir: root,
	})
	if err != nil {
		log.Fatalf("fastface.New: %v", err)
	}
	defer ff.Close()

	srv := &server{ff: ff}
	http.HandleFunc("/embed", srv.handleEmbed)
	http.HandleFunc("/match", srv.handleMatch)
	http.HandleFunc("/health", srv.handleHealth)
	log.Printf("FastFace httpd listening on %s", *listen)
	log.Fatal(http.ListenAndServe(*listen, nil))
}
