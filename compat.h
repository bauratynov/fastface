#ifndef FASTFACE_COMPAT_H
#define FASTFACE_COMPAT_H

/* Cross-platform compatibility shims.
 *
 * The engine was originally developed on Windows (MinGW) and used the
 * MS-runtime functions _aligned_malloc / _aligned_free / _setmode /
 * _O_BINARY / _fileno directly. This header provides the same names on
 * Linux / macOS via POSIX equivalents so the source compiles everywhere.
 */

#include <stddef.h>

#ifdef _WIN32
  #include <windows.h>
  #include <io.h>
  #include <fcntl.h>
#else
  #include <stdlib.h>
  #include <unistd.h>
  #include <stdio.h>

  /* _aligned_malloc(size, align) -> posix_memalign */
  static inline void* _aligned_malloc(size_t size, size_t align) {
    /* POSIX requires alignment to be a power of 2 and multiple of sizeof(void*),
     * and size to be a multiple of alignment. */
    if (align < sizeof(void*)) align = sizeof(void*);
    size = (size + align - 1) & ~(align - 1);
    void* p = NULL;
    if (posix_memalign(&p, align, size) != 0) return NULL;
    return p;
  }
  #define _aligned_free free

  /* _setmode/_O_BINARY are Windows-only. Linux file descriptors are always
   * "binary" — make these no-ops. */
  #define _setmode(fd, mode) (0)
  #define _O_BINARY 0
  #define _fileno(f) fileno(f)
#endif

#endif /* FASTFACE_COMPAT_H */
