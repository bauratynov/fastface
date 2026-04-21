// Session 7: FFW2 binary loader + op dispatcher skeleton.
//
// FFW2 format (from prepare_weights_v2.py):
//   'FFW2' magic, u32 n_ops
//   For each op: u8 op_type, then type-specific payload:
//     OP_CONV (1):  H×6u16 [Cin, Cout, Kh, Kw, stride, pad]  i8[Cout*Cin*Kh*Kw] f32[Cout] scales
//     OP_BN (2):    u16 size, f32[size] scale, f32[size] offset
//     OP_PRELU (3): u16 size, f32[size] slope
//     OP_ADD (4):   no payload
//     OP_GEMM (5):  u32 N, u32 K, i8[N*K] f32[N] scales f32[N] bias
//     OP_FLATTEN (6): no payload
//     OP_SAVE_ID (7): no payload

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <windows.h>

#define OP_CONV 1
#define OP_BN 2
#define OP_PRELU 3
#define OP_ADD 4
#define OP_GEMM 5
#define OP_FLATTEN 6
#define OP_SAVE_ID 7
#define OP_BLOCK_START 8

typedef struct {
    uint8_t type;
    // Conv
    uint16_t Cin, Cout, Kh, Kw, stride, pad;
    const int8_t*  conv_w;       // [Cout * Cin * Kh * Kw] int8 (row-major)
    const float*   conv_scales;  // [Cout]
    const float*   conv_bias;    // [Cout] — present in FFW3+
    // BN
    const float*   bn_scale;     // [Cout]
    const float*   bn_offset;    // [Cout]
    uint16_t bn_size;
    // PRelu
    const float*   prelu_slope;
    uint16_t prelu_size;
    // Gemm
    uint32_t N, K;
    const int8_t*  gemm_w;
    const float*   gemm_scales;
    const float*   gemm_bias;
} Op;

typedef struct {
    void* data;          // mmap buffer
    size_t size;
    uint32_t n_ops;
    Op*      ops;        // array of n_ops
    uint8_t  version;    // 2, 3, or 4 — S37: FFW4 = Gemm weights pre-folded with per-channel act scale
} FFW2;


static const uint8_t* read_u8(const uint8_t* p, uint8_t* out) { *out = *p; return p + 1; }
static const uint8_t* read_u16(const uint8_t* p, uint16_t* out) { memcpy(out, p, 2); return p + 2; }
static const uint8_t* read_u32(const uint8_t* p, uint32_t* out) { memcpy(out, p, 4); return p + 4; }


int ffw2_load(const char* path, FFW2* out) {
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "open fail\n"); return -1; }
    LARGE_INTEGER sz; GetFileSizeEx(h, &sz);
    HANDLE m = CreateFileMappingA(h, NULL, PAGE_READONLY, 0, 0, NULL);
    void* data = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!data) { fprintf(stderr, "mmap fail\n"); return -2; }
    out->data = data; out->size = (size_t)sz.QuadPart;

    const uint8_t* p = (const uint8_t*)data;
    int version;
    if      (memcmp(p, "FFW4", 4) == 0) version = 4;
    else if (memcmp(p, "FFW3", 4) == 0) version = 3;
    else if (memcmp(p, "FFW2", 4) == 0) version = 2;
    else { fprintf(stderr, "bad magic\n"); return -3; }
    out->version = (uint8_t)version;
    p += 4;
    uint32_t n_ops;
    p = read_u32(p, &n_ops);
    out->n_ops = n_ops;
    out->ops = (Op*)calloc(n_ops, sizeof(Op));

    for (uint32_t i = 0; i < n_ops; i++) {
        Op* op = &out->ops[i];
        p = read_u8(p, &op->type);
        switch (op->type) {
            case OP_CONV: {
                p = read_u16(p, &op->Cin);
                p = read_u16(p, &op->Cout);
                p = read_u16(p, &op->Kh);
                p = read_u16(p, &op->Kw);
                p = read_u16(p, &op->stride);
                p = read_u16(p, &op->pad);
                size_t w_elems = (size_t)op->Cout * op->Cin * op->Kh * op->Kw;
                op->conv_w = (const int8_t*)p; p += w_elems;
                op->conv_scales = (const float*)p; p += op->Cout * sizeof(float);
                if (version >= 3) {
                    op->conv_bias = (const float*)p; p += op->Cout * sizeof(float);
                } else {
                    op->conv_bias = NULL;
                }
                break;
            }
            case OP_BN: {
                p = read_u16(p, &op->bn_size);
                op->bn_scale = (const float*)p; p += op->bn_size * sizeof(float);
                op->bn_offset = (const float*)p; p += op->bn_size * sizeof(float);
                break;
            }
            case OP_PRELU: {
                p = read_u16(p, &op->prelu_size);
                op->prelu_slope = (const float*)p; p += op->prelu_size * sizeof(float);
                break;
            }
            case OP_ADD:
            case OP_FLATTEN:
            case OP_SAVE_ID:
            case OP_BLOCK_START:
                break;
            case OP_GEMM: {
                p = read_u32(p, &op->N);
                p = read_u32(p, &op->K);
                size_t w_elems = (size_t)op->N * op->K;
                op->gemm_w = (const int8_t*)p; p += w_elems;
                op->gemm_scales = (const float*)p; p += op->N * sizeof(float);
                op->gemm_bias   = (const float*)p; p += op->N * sizeof(float);
                break;
            }
            default:
                fprintf(stderr, "bad op type %d at op %u\n", op->type, i);
                return -4;
        }
    }
    return 0;
}


#ifndef FFW2_NOMAIN
int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "../models/w600k_r50_ffw2.bin";
    FFW2 m = {0};
    if (ffw2_load(path, &m) != 0) return 1;

    printf("Loaded %s\n", path);
    printf("  Total bytes: %.1f MB\n", (double)m.size / 1024 / 1024);
    printf("  Ops: %u\n\n", m.n_ops);

    // Count by type
    int cnt[8] = {0};
    for (uint32_t i = 0; i < m.n_ops; i++) cnt[m.ops[i].type]++;
    printf("  Conv: %d\n  BN: %d\n  PRelu: %d\n  Add: %d\n  Gemm: %d\n  Flatten: %d\n  Save_Id: %d\n\n",
           cnt[1], cnt[2], cnt[3], cnt[4], cnt[5], cnt[6], cnt[7]);

    // Dump first 12 ops with shapes
    printf("First 12 ops:\n");
    for (uint32_t i = 0; i < 12 && i < m.n_ops; i++) {
        const Op* op = &m.ops[i];
        switch (op->type) {
            case OP_CONV:
                printf("  [%2u] Conv Cin=%u Cout=%u k=%ux%u s=%u p=%u  scale[0]=%.4f scale[-1]=%.4f\n",
                       i, op->Cin, op->Cout, op->Kh, op->Kw, op->stride, op->pad,
                       op->conv_scales[0], op->conv_scales[op->Cout - 1]);
                break;
            case OP_BN:
                printf("  [%2u] BN   size=%u  scale[0]=%.4f offset[0]=%.4f\n",
                       i, op->bn_size, op->bn_scale[0], op->bn_offset[0]);
                break;
            case OP_PRELU:
                printf("  [%2u] PRelu size=%u slope[0]=%.4f\n",
                       i, op->prelu_size, op->prelu_slope[0]);
                break;
            case OP_ADD:
                printf("  [%2u] Add\n", i); break;
            case OP_GEMM:
                printf("  [%2u] Gemm N=%u K=%u\n", i, op->N, op->K); break;
            case OP_FLATTEN:
                printf("  [%2u] Flatten\n", i); break;
            case OP_SAVE_ID:
                printf("  [%2u] Save_Identity\n", i); break;
        }
    }

    return 0;
}
#endif
