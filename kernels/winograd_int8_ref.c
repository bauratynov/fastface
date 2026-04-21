// S34 phase 1 — INT8 Winograd F(2,3) scalar reference.
//
// Math: F(2, 3) Winograd transforms a 3x3 conv over a 4x4 input tile into a
// 2x2 output tile via:
//   Y = A^T [ (G g G^T) (*) (B^T d B) ] A
// where (*) is element-wise multiply over the 4x4 transform grid, summed
// across input channels.
//
// Standard F(2,3) matrices:
//   B^T = [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]
//   G   = [[1, 0, 0], [1/2, 1/2, 1/2], [1/2, -1/2, 1/2], [0, 0, 1]]
//   A^T = [[1, 1, 1, 0], [0, 1, -1, -1]]
//
// To keep intermediate arithmetic in integers we use G' = 2G:
//   G'  = [[2, 0, 0], [1, 1, 1], [1, -1, 1], [0, 0, 2]]
// Then (G' g G'^T) = 4 (G g G^T). Output Y' = A^T ( (G' g G'^T) (*) (B^T d B) ) A
// equals 4 Y. Divide by 4 at the dequant step by folding into out_scale.
//
// Ranges (signed):
//   B^T d B: entries of B^T are {-1, 0, 1}, so |v| <= 4 * 127 = 508. int16 ok.
//   G' g G'^T: entries of G' max |2|, so |u| <= 4 * 2 * 127 = 1016 (looser bound),
//              realistic with sign cancellations. int16 ok.
//   Elementwise v * u: <= 508 * 1016 = 516K per element. int32 ok.
//   Sum over Cin (up to 512): <= 516K * 512 = 264M. int32 ok (max 2.1B).
//
// Scale accounting: after Winograd we hold 4 * (fp conv result / (in_scale * w_scale)).
// Fused epilogue should multiply by (in_scale * w_scale / 4). Caller handles.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Weight transform: u = G' g G'^T for each (oc, ci). Stored as [4, 4, Cout, Cin].
// g is laid out as [Cout, Cin, 3, 3] int8.
void winograd_int8_transform_weights_ref(
    const int8_t* g,
    int Cout, int Cin,
    int16_t* u)
{
    // G' * row: row is 3 entries. Result is 4 entries.
    //   G' row 0 = [2, 0, 0]        -> 2*a
    //   G' row 1 = [1, 1, 1]        -> a + b + c
    //   G' row 2 = [1, -1, 1]       -> a - b + c
    //   G' row 3 = [0, 0, 2]        -> 2*c

    for (int oc = 0; oc < Cout; oc++) {
        for (int ci = 0; ci < Cin; ci++) {
            // Fetch 3x3 weight
            int16_t a[3][3];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    a[i][j] = (int16_t)g[(((size_t)oc * Cin + ci) * 3 + i) * 3 + j];

            // Step 1: temp = G' * a  (4x3)
            int16_t tmp[4][3];
            for (int j = 0; j < 3; j++) {
                tmp[0][j] = 2 * a[0][j];
                tmp[1][j] = a[0][j] + a[1][j] + a[2][j];
                tmp[2][j] = a[0][j] - a[1][j] + a[2][j];
                tmp[3][j] = 2 * a[2][j];
            }
            // Step 2: out = tmp * G'^T  (4x4). G'^T col transforms:
            //   col 0: [2, 0, 0]      -> 2*tmp[,0]
            //   col 1: [1, 1, 1]      -> tmp[,0] + tmp[,1] + tmp[,2]
            //   col 2: [1, -1, 1]     -> tmp[,0] - tmp[,1] + tmp[,2]
            //   col 3: [0, 0, 2]      -> 2*tmp[,2]
            for (int i = 0; i < 4; i++) {
                int16_t u0 = 2 * tmp[i][0];
                int16_t u1 = tmp[i][0] + tmp[i][1] + tmp[i][2];
                int16_t u2 = tmp[i][0] - tmp[i][1] + tmp[i][2];
                int16_t u3 = 2 * tmp[i][2];
                u[((size_t)i * 4 + 0) * Cout * Cin + (size_t)oc * Cin + ci] = u0;
                u[((size_t)i * 4 + 1) * Cout * Cin + (size_t)oc * Cin + ci] = u1;
                u[((size_t)i * 4 + 2) * Cout * Cin + (size_t)oc * Cin + ci] = u2;
                u[((size_t)i * 4 + 3) * Cout * Cin + (size_t)oc * Cin + ci] = u3;
            }
        }
    }
}

// Input transform on a 4x4 tile: v = B^T d B. Each d[i][j] is int8.
// B^T row transforms: [1,0,-1,0], [0,1,1,0], [0,-1,1,0], [0,1,0,-1]
static void winograd_int8_input_tile(const int16_t d[4][4], int16_t v[4][4]) {
    int16_t tmp[4][4];
    for (int j = 0; j < 4; j++) {
        tmp[0][j] = d[0][j] - d[2][j];
        tmp[1][j] = d[1][j] + d[2][j];
        tmp[2][j] = -d[1][j] + d[2][j];
        tmp[3][j] = d[1][j] - d[3][j];
    }
    for (int i = 0; i < 4; i++) {
        v[i][0] = tmp[i][0] - tmp[i][2];
        v[i][1] = tmp[i][1] + tmp[i][2];
        v[i][2] = -tmp[i][1] + tmp[i][2];
        v[i][3] = tmp[i][1] - tmp[i][3];
    }
}

// Output transform: Y = A^T M A where M is 4x4 int32, Y is 2x2 int32.
// A^T rows: [1,1,1,0], [0,1,-1,-1]
static void winograd_int8_output_tile(const int32_t M[4][4], int32_t y[2][2]) {
    int32_t tmp[2][4];
    for (int j = 0; j < 4; j++) {
        tmp[0][j] = M[0][j] + M[1][j] + M[2][j];
        tmp[1][j] = M[1][j] - M[2][j] - M[3][j];
    }
    for (int i = 0; i < 2; i++) {
        y[i][0] = tmp[i][0] + tmp[i][1] + tmp[i][2];
        y[i][1] = tmp[i][1] - tmp[i][2] - tmp[i][3];
    }
}

// Reference Winograd F(2,3) conv. Input NHWC [H, W, Cin], padding 1 stride 1.
// Output NHWC [H, W, Cout] int32 (pre-dequant). 4x Y (see scale note at top).
// H and W must be even (we do 2x2 output tiles).
void fastface_conv2d_i8_winograd_f23_ref(
    const int8_t* input,
    int Cin, int H, int W,
    int Cout,
    const int16_t* u_weights,  // [4, 4, Cout, Cin]
    int32_t* output)
{
    int H_out = H, W_out = W;  // stride 1 pad 1

    for (int oh = 0; oh < H_out; oh += 2) {
        for (int ow = 0; ow < W_out; ow += 2) {
            // For each output 2x2 tile, the input 4x4 tile starts at (oh-1, ow-1) with pad 1.
            int ih0 = oh - 1, iw0 = ow - 1;

            // Per-oc accumulator for this tile: 4x4 int32 grid
            for (int oc = 0; oc < Cout; oc++) {
                int32_t M[4][4];
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        M[i][j] = 0;

                for (int ci = 0; ci < Cin; ci++) {
                    // Fetch input 4x4 tile for this ci (with zero padding)
                    int16_t d[4][4];
                    for (int ti = 0; ti < 4; ti++) {
                        for (int tj = 0; tj < 4; tj++) {
                            int ih = ih0 + ti, iw = iw0 + tj;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                d[ti][tj] = (int16_t)input[((size_t)ih * W + iw) * Cin + ci];
                            else
                                d[ti][tj] = 0;
                        }
                    }
                    int16_t v[4][4];
                    winograd_int8_input_tile(d, v);

                    // Element-wise multiply with this ci's weight slice and accumulate
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            int16_t u_val = u_weights[((size_t)i * 4 + j) * Cout * Cin + (size_t)oc * Cin + ci];
                            M[i][j] += (int32_t)v[i][j] * (int32_t)u_val;
                        }
                    }
                }

                // Output transform for this oc: 4x4 -> 2x2 int32
                int32_t y[2][2];
                winograd_int8_output_tile(M, y);

                // Scatter 2x2 into output NHWC
                for (int di = 0; di < 2; di++) {
                    int oh_out = oh + di;
                    if (oh_out >= H_out) continue;
                    for (int dj = 0; dj < 2; dj++) {
                        int ow_out = ow + dj;
                        if (ow_out >= W_out) continue;
                        output[((size_t)oh_out * W_out + ow_out) * Cout + oc] = y[di][dj];
                    }
                }
            }
        }
    }
}

#ifdef WINOGRAD_I8_REF_TEST

// Direct reference Conv (baseline to validate against).
static void direct_conv_i8_ref(
    const int8_t* input, int Cin, int H, int W,
    int Cout, const int8_t* g,
    int32_t* output)
{
    int H_out = H, W_out = W;  // stride 1 pad 1
    for (int oh = 0; oh < H_out; oh++) {
        for (int ow = 0; ow < W_out; ow++) {
            for (int oc = 0; oc < Cout; oc++) {
                int32_t acc = 0;
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int ih = oh + kh - 1, iw = ow + kw - 1;
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                        for (int ci = 0; ci < Cin; ci++) {
                            acc += (int32_t)input[((size_t)ih * W + iw) * Cin + ci]
                                 * (int32_t)g[(((size_t)oc * Cin + ci) * 3 + kh) * 3 + kw];
                        }
                    }
                }
                output[((size_t)oh * W_out + ow) * Cout + oc] = acc;
            }
        }
    }
}

int main(int argc, char** argv) {
    int H = 8, W = 8, Cin = 16, Cout = 8;
    if (argc >= 5) {
        H = atoi(argv[1]); W = atoi(argv[2]);
        Cin = atoi(argv[3]); Cout = atoi(argv[4]);
    }

    size_t n_in = (size_t)H * W * Cin;
    size_t n_w  = (size_t)Cout * Cin * 3 * 3;
    size_t n_out = (size_t)H * W * Cout;

    int8_t* input = (int8_t*)malloc(n_in);
    int8_t* weight = (int8_t*)malloc(n_w);
    int16_t* u_wino = (int16_t*)malloc((size_t)16 * Cout * Cin * sizeof(int16_t));
    int32_t* out_direct = (int32_t*)malloc(n_out * sizeof(int32_t));
    int32_t* out_wino   = (int32_t*)malloc(n_out * sizeof(int32_t));

    srand(42);
    // Full int8 range [-127, 127] to stress accumulator width
    for (size_t i = 0; i < n_in; i++) input[i]  = (int8_t)((rand() % 255) - 127);
    for (size_t i = 0; i < n_w;  i++) weight[i] = (int8_t)((rand() % 255) - 127);

    direct_conv_i8_ref(input, Cin, H, W, Cout, weight, out_direct);

    winograd_int8_transform_weights_ref(weight, Cout, Cin, u_wino);
    fastface_conv2d_i8_winograd_f23_ref(input, Cin, H, W, Cout, u_wino, out_wino);

    // Winograd output is 4x direct (because G' = 2G, scale factor 4).
    int32_t max_err = 0;
    size_t n_nonexact = 0;
    for (size_t i = 0; i < n_out; i++) {
        int32_t expected = out_direct[i] * 4;  // Winograd = 4 * direct
        int32_t err = out_wino[i] - expected;
        if (err < 0) err = -err;
        if (err > max_err) max_err = err;
        if (err != 0) n_nonexact++;
    }
    printf("Winograd-Direct comparison (H=%d W=%d Cin=%d Cout=%d):\n", H, W, Cin, Cout);
    printf("  total elements: %zu\n", n_out);
    printf("  non-exact elements: %zu (%.1f%%)\n", n_nonexact, 100.0 * n_nonexact / n_out);
    printf("  max abs error: %d\n", max_err);
    printf("  sample wino/expected: [%d] %d vs %d,  [%d] %d vs %d,  [%d] %d vs %d\n",
           0, out_wino[0], out_direct[0] * 4,
           50, out_wino[50], out_direct[50] * 4,
           100, out_wino[100], out_direct[100] * 4);

    if (max_err == 0)
        printf("PASS: INT8 Winograd F(2,3) is integer-exact vs direct conv.\n");
    else
        printf("FAIL: max error %d (expected 0 for integer-exact Winograd).\n", max_err);

    free(input); free(weight); free(u_wino); free(out_direct); free(out_wino);
    return max_err == 0 ? 0 : 1;
}
#endif
