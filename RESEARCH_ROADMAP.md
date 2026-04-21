# FastFace research roadmap (post-S17 victory)

Essential reading list and mathematical foundations for the next 3-4 sessions.

## Foundational papers

### Winograd for CNN
1. **Lavin, Gray (2016). "Fast Algorithms for Convolutional Neural Networks."** CVPR 2016.
   arxiv: https://arxiv.org/abs/1509.09308
   — The paper that introduced F(m, r) to deep learning. Transform matrices for F(2,3), F(4,3). Savings analysis.

2. **Barbara, Szabo, Koishybayeva (2020). "Error Analysis and Improving the Accuracy of Winograd Convolution for DNNs."** ACM Trans. Math. Software.
   https://dl.acm.org/doi/10.1145/3412380
   — How FP error grows with tile size. Symmetric interpolation points to reduce error.

3. **Vincent, Stephano (2017). "On Improving the Numerical Stability of Winograd Convolutions."**
   OpenReview: https://openreview.net/forum?id=H1ZaRZVKg
   — Careful transform matrix choice.

4. **Lavin (2015). `wincnn` — Python Winograd generator.**
   https://github.com/andravin/wincnn
   — Code to generate exact F(m, r) matrices via Cook-Toom. **Use this to get F(4,3) matrices for Session 19.**

### GEMM optimization
5. **Salykova (2024). "Advanced Matrix Multiplication Optimization on Modern Multi-Core Processors."**
   https://salykova.github.io/matmul-cpu
   — Modern tutorial. MR=6, NR=16 for AVX2 FP32. Cache blocking MC, NC, KC.

6. **Van Zee, Smith, Goto (2013). "The BLIS Framework: Experiments in Portability."**
   https://www.cs.utexas.edu/~flame/pubs/sc13.pdf
   — The definitive GEMM architecture. GotoBLAS lineage.

7. **BLIS HardwareSupport docs.**
   https://github.com/flame/blis/blob/master/docs/HardwareSupport.md
   — Confirms MR=6 NR=16 on AVX2 Haswell/Skylake/later.

### INT8 quantization
8. **Migacz (2017). "8-bit Inference with TensorRT."** GTC San Jose.
   https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
   — The original KL-divergence calibration algorithm. Pseudocode, histograms, threshold search.

9. **Wu et al. (2020). "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation."**
   arxiv: https://arxiv.org/abs/2004.09602
   — Modern survey: per-channel vs per-tensor, symmetric vs asymmetric, PTQ vs QAT.

10. **Mori et al. (2024). "Wino Vidi Vici: Conquering Numerical Instability of 8-bit Winograd Convolution."** WACV 2024.
    — Why you shouldn't naively combine INT8 with Winograd. If you want both, this is the recipe.

## Key math cheatsheet

### Winograd F(m, r) complexity
- 1D multiplications: **m + r − 1** vs standard m·r
- 2D savings: ((m·r) / (m+r−1))²
- For 3×3 kernel: F(2,3)=2.25×, F(4,3)=4.0×, F(6,3)=5.06×
- Interpolation points: need **m+r−2** points (Lagrange polynomial degree)

### F(4, 3) matrices (from wincnn with points (0,1,−1,2,−2))
```
AT = [ 1  1   1   1    1    0 ]
     [ 0  1  −1   2   −2    0 ]
     [ 0  1   1   4    4    0 ]
     [ 0  1  −1   8   −8    1 ]

# G contains fractions like ±1/4, ±1/6, ±1/24
# BT contains values {4, −5, 1/2, etc.}
```

### BLIS GEMM register budget (AVX2 FP32)
- 16 YMM regs × 8 fp32 lanes
- MR × NR output tile = 6 × 16
- C accumulators: MR × (NR/8) = 12 regs
- B panel: 2 regs (NR/8)
- A broadcasts: 1-2 regs
- Utilization: 12/16 = 75% (vs our 8/16 = 50% with MR=4)

### KL calibration pseudocode
```
for each layer L:
    H = histogram of activations over calibration set (2048 bins)
    best_T, best_KL = null, infinity
    for T in range(128, 2048):
        # Reference distribution: clip H to [0, T), sum tail into H[T-1]
        P = H[0:T].copy(); P[T-1] += sum(H[T:])
        P = P / sum(P)  # normalize
        # Quantized distribution: bin [0, T) into 128 int8 bins, then expand back
        Q = bin_into_128(H[0:T])
        Q_expanded = expand_to_T_bins(Q)
        Q_expanded = Q_expanded / sum(Q_expanded)
        # Only consider non-zero mass
        KL = sum(P[i] * log(P[i] / Q_expanded[i]) for i where P[i] > 0)
        if KL < best_KL:
            best_T, best_KL = T, KL
    scale_L = (best_T * bin_width) / 127
```

## Session sequence to apply

| session | topic | source | expected outcome |
|---|---|---|---|
| S18 | Upgrade GEMM to MR=6, NR=16 | BLIS HardwareSupport + Salykova | FP32: 29 → 25-27 ms |
| S19 | Winograd F(4, 3) with wincnn-generated matrices | Lavin 2016 + wincnn | FP32: 25 → 22-24 ms |
| S20 | KL calibration pipeline + full INT8 path | Migacz 2017 | INT8: 10-13 ms @ cos-sim 0.95+ |
| S21 | Optional: F(6, 3) for max FP32 savings | Barbara 2020 | FP32: 18-20 ms |

## Open questions to answer next time

1. How far can we push MR on i7-13700? Raptor Lake has more YMM/ZMM? (no ZMM in consumer, but 16 YMM regs same as Haswell).
2. F(4, 3) on INT8 — is it worth attempting or skip entirely per Mori 2024?
3. Can we combine our Winograd F(2, 3) with MR=6 GEMM without a full rewrite?
4. How does ORT's im2col + MLAS GEMM actually achieve 31.76 ms? Is their packed GEMM MR=6 NR=16 too?
