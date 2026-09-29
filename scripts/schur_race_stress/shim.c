#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
void ncgesv(int *n, int *nrhs, void *a, int *lda, int *piv, void *b, int *ldb, int *info) {
    cgesv_(n, nrhs, (__LAPACK_float_complex *)a, lda, piv, (__LAPACK_float_complex *)b, ldb, info);
}
void nzgesv(int *n, int *nrhs, void *a, int *lda, int *piv, void *b, int *ldb, int *info) {
    zgesv_(n, nrhs, (__LAPACK_double_complex *)a, lda, piv, (__LAPACK_double_complex *)b, ldb, info);
}
