# GitHub Topics (20)

Per post-S131 mandate, apply these topics after `gh repo create`:

```
face-recognition
arcface
int8-quantization
cpu-inference
avx-vnni
face-verification
insightface
vnni
simd
c99
onnx
biometrics
edge-ai
neural-network
deep-learning
computer-vision
low-latency
embedded-ai
production-ready
open-source
```

## gh CLI incantation

```bash
gh repo edit --add-topic face-recognition \
             --add-topic arcface \
             --add-topic int8-quantization \
             --add-topic cpu-inference \
             --add-topic avx-vnni \
             --add-topic face-verification \
             --add-topic insightface \
             --add-topic vnni \
             --add-topic simd \
             --add-topic c99 \
             --add-topic onnx \
             --add-topic biometrics \
             --add-topic edge-ai \
             --add-topic neural-network \
             --add-topic deep-learning \
             --add-topic computer-vision \
             --add-topic low-latency \
             --add-topic embedded-ai \
             --add-topic production-ready \
             --add-topic open-source
```

## Rationale per topic

| topic | why |
|---|---|
| face-recognition | primary discovery term |
| arcface | specific model discovery |
| int8-quantization | quantization crowd discovery |
| cpu-inference | differentiator vs GPU-first projects |
| avx-vnni | low-level SIMD crowd |
| face-verification | LFW / verification protocol crowd |
| insightface | ecosystem affinity (model source) |
| vnni | VNNI instruction discovery |
| simd | general SIMD audience |
| c99 | pure-C fans |
| onnx | ONNX model origin signal |
| biometrics | broader biometrics domain |
| edge-ai | edge deployment crowd |
| neural-network | generic discovery fallback |
| deep-learning | broader DL audience |
| computer-vision | CV community |
| low-latency | latency-sensitive production crowd |
| embedded-ai | embedded board / SoC vendor audience |
| production-ready | signal of readiness (not research-only) |
| open-source | general FOSS flag |
