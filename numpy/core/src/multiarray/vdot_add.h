#ifndef _NPY_VDOT_ADD_H_
#define _NPY_VDOT_ADD_H_

#include "common.h"

NPY_NO_EXPORT void
SHORT_vdot_add(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
INT_vdot_add(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
LONG_vdot_add(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
FLOAT_vdot_add(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
DOUBLE_vdot_add(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
LONGDOUBLE_vdot_add(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
CFLOAT_vdot_add(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
CDOUBLE_vdot_add(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

NPY_NO_EXPORT void
CLONGDOUBLE_vdot_add(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

//NPY_NO_EXPORT void
//OBJECT_vdot_add(char *, npy_intp, char *, npy_intp, char *, npy_intp, void *);

#endif
