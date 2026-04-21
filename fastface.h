/* fastface.h -- C API for the FastFace INT8 face embedding engine.
 *
 * Usage:
 *     FastFace* ff = fastface_create("models/w600k_r50_ffw4.bin");
 *     if (!ff) return -1;
 *     float input[3*112*112];    // HWC NHWC, [-1, 1]
 *     float embedding[512];
 *     fastface_embed(ff, input, embedding);
 *     fastface_destroy(ff);
 *
 * Thread-safety: one context per thread; call fastface_create from each
 * thread that needs concurrent inference.
 *
 * Input layout:  HWC contiguous float32, 3*112*112 = 37632 floats, range [-1,1].
 * Output layout: raw (not L2-normalized) float32 embedding, 512 floats.
 *
 * Matches the --server protocol byte-for-byte.
 */

#ifndef FASTFACE_H
#define FASTFACE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FastFace FastFace;

/* Create a context from FFW4 weights file. Returns NULL on failure.
 * The file is mmaped; do not delete it while the context lives. */
FastFace* fastface_create(const char* ffw4_path);

/* Run one forward pass. Caller provides 3*112*112 input fp32 and 512
 * output fp32. Returns 0 on success, non-zero on error. */
int fastface_embed(FastFace* ff, const float* input_hwc, float* out_emb);

/* Free all context resources. */
void fastface_destroy(FastFace* ff);

/* Constants (informational) */
#define FASTFACE_INPUT_N  37632   /* 3 * 112 * 112 */
#define FASTFACE_OUTPUT_N 512

#ifdef __cplusplus
}
#endif

#endif /* FASTFACE_H */
