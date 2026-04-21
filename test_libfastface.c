#include <stdio.h>
#include <stdlib.h>
#include "fastface.h"

int main(void) {
    FastFace* ff = fastface_create("models/w600k_r50_ffw4.bin");
    if (!ff) { fprintf(stderr, "create failed\n"); return 1; }
    FILE* f = fopen("tests/golden_input.bin", "rb");
    if (!f) { fprintf(stderr, "cannot open golden input\n"); return 2; }
    float input[FASTFACE_INPUT_N];
    fread(input, sizeof(float), FASTFACE_INPUT_N, f);
    fclose(f);
    float emb[FASTFACE_OUTPUT_N];
    if (fastface_embed(ff, input, emb) != 0) { fprintf(stderr, "embed failed\n"); return 3; }
    printf("emb[0..4] = %g %g %g %g %g\n", emb[0], emb[1], emb[2], emb[3], emb[4]);
    fastface_destroy(ff);
    return 0;
}
