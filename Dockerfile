# FastFace — reproducible build container.
#
# Build:  docker build -t fastface:latest .
# Test:   docker run --rm fastface:latest make test
# Bench:  docker run --rm fastface:latest ./fastface_int8.exe models/w600k_r50_ffw4.bin
# Shell:  docker run --rm -it fastface:latest bash
#
# Note: GitHub-hosted / generic-cloud runners without AVX-VNNI still build
# and pass regression tests (accuracy bit-exact vs golden), but speed
# numbers will be slower than README's i7-13700 figures.

FROM ubuntu:24.04 AS builder

# Build toolchain + Python runtime for calibration scripts and Python SDK
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc-13 make ca-certificates \
        python3 python3-numpy python3-pil \
        golang-1.22 \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-13
ENV PATH="/usr/lib/go-1.22/bin:${PATH}"

WORKDIR /opt/fastface

# Copy only what's needed for build (speeds up docker cache)
COPY Makefile fastface.h *.c *.py ./
COPY kernels/ ./kernels/
COPY models/*.bin ./models/
COPY tests/ ./tests/
COPY go/ ./go/

# Strip "fastface_int8.exe" .exe suffix in Makefile for Linux? No --
# Makefile uses .exe as a literal suffix; works fine on Linux (file just
# has a non-standard name, still executable).
RUN make CC=gcc-13 AR=ar all

# Regression gate: image build fails if regression test fails
RUN make PYTHON=python3 test

# Go SDK unit test (skip examples/ which need internet)
RUN cd go/fastface && go test ./...

# --- Final stage: lean runtime image ---
FROM ubuntu:24.04 AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-numpy python3-pil libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/fastface

COPY --from=builder /opt/fastface/fastface_int8.exe .
COPY --from=builder /opt/fastface/fastface_int8_batched.exe .
COPY --from=builder /opt/fastface/libfastface.a .
COPY --from=builder /opt/fastface/fastface.h .
COPY --from=builder /opt/fastface/fastface.py .
COPY --from=builder /opt/fastface/face_match.py .
COPY --from=builder /opt/fastface/models/w600k_r50_ffw4.bin models/
COPY --from=builder /opt/fastface/models/op_scales.bin models/
COPY --from=builder /opt/fastface/models/op_scales_v2.bin models/
COPY --from=builder /opt/fastface/tests/ ./tests/

# Validate the runtime layer still works
RUN ./fastface_int8.exe models/w600k_r50_ffw4.bin --in tests/golden_input.bin \
                       --out /tmp/emb.bin && \
    test -s /tmp/emb.bin

ENTRYPOINT []
CMD ["./fastface_int8.exe", "models/w600k_r50_ffw4.bin"]
