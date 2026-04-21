// findface — example of using the Go FastFace SDK to find the closest
// face in a directory given a query face. Stdlib-only, no cgo.
//
// Usage:
//     go run main.go <query.jpg> <directory>
package main

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	fastface "github.com/bauratynov/fastface/go/fastface"
)

const FaceSize = 112

// centerCropResize does a 150-px center crop (shifted up 10px) then
// bilinearly resamples to 112x112 and returns HWC fp32 in [-1, 1].
// Zero-dependency (stdlib only) but not optimized.
func preprocess(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}
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
			// Bilinear sample
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
	return arr, nil
}

type match struct {
	path string
	sim  float32
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
	if len(os.Args) != 3 {
		fmt.Fprintf(os.Stderr, "Usage: %s <query.jpg> <directory>\n", os.Args[0])
		os.Exit(1)
	}
	queryPath, dirPath := os.Args[1], os.Args[2]

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

	qArr, err := preprocess(queryPath)
	if err != nil {
		log.Fatalf("query preprocess: %v", err)
	}
	qEmb, err := ff.Embed(qArr)
	if err != nil {
		log.Fatalf("query embed: %v", err)
	}

	var matches []match
	err = filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}
		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
			return nil
		}
		arr, err := preprocess(path)
		if err != nil {
			return nil
		}
		emb, err := ff.Embed(arr)
		if err != nil {
			return err
		}
		matches = append(matches, match{path: path, sim: fastface.CosSim(qEmb, emb)})
		return nil
	})
	if err != nil {
		log.Fatalf("walk: %v", err)
	}

	sort.Slice(matches, func(i, j int) bool { return matches[i].sim > matches[j].sim })

	fmt.Printf("\nTop 5 matches for %s:\n", queryPath)
	n := 5
	if n > len(matches) {
		n = len(matches)
	}
	for i := 0; i < n; i++ {
		verdict := "different"
		if matches[i].sim >= 0.20 {
			verdict = "SAME"
		}
		fmt.Printf("  %+.4f  %-9s  %s\n", matches[i].sim, verdict, matches[i].path)
	}
}
