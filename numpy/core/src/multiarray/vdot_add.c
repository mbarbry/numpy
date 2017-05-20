#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include "common.h"
#include "vdot_add.h"
//#include "npy_cblas.h"


/*
 * vdot_add
 * perform a dot product and adding a scalr to the array
 * All data is assumed aligned.
 */
NPY_NO_EXPORT void
SHORT_vdot_add(char *ip1, npy_intp is1, char *ip2, npy_intp is2,
    char *op, npy_intp n, void *NPY_UNUSED(ignore))
{
    short sum = (short)0;
    const short ip2r = ((short *)ip2)[0];
    npy_intp i;


    for (i = 0; i < n; i++, ip1 += is1) {
        const short ip1r = ((short *)ip1)[0] + ip2r;

        sum += ip1r * ip1r;
    }
    ((short *)op)[0] = sum;
}

NPY_NO_EXPORT void
INT_vdot_add(char *ip1, npy_intp is1, char *ip2, npy_intp is2,
    char *op, npy_intp n, void *NPY_UNUSED(ignore))
{
    int sum = (int)0;
    const int ip2r = ((int *)ip2)[0];
    npy_intp i;


    for (i = 0; i < n; i++, ip1 += is1) {
        const int ip1r = ((int *)ip1)[0] + ip2r;

        sum += ip1r * ip1r;
    }
    ((int *)op)[0] = sum;
}

NPY_NO_EXPORT void
LONG_vdot_add(char *ip1, npy_intp is1, char *ip2, npy_intp is2,
    char *op, npy_intp n, void *NPY_UNUSED(ignore))
{
    long sum = (long)0;
    const long ip2r = ((long *)ip2)[0];
    npy_intp i;


    for (i = 0; i < n; i++, ip1 += is1) {
        const long ip1r = ((long *)ip1)[0] + ip2r;

        sum += ip1r * ip1r;
    }
    ((long *)op)[0] = sum;
}

NPY_NO_EXPORT void
FLOAT_vdot_add(char *ip1, npy_intp is1, char *ip2, npy_intp is2,
    char *op, npy_intp n, void *NPY_UNUSED(ignore))
{
    float sum = (float)0.0;
    const float ip2r = ((float *)ip2)[0];
    npy_intp i;


    for (i = 0; i < n; i++, ip1 += is1) {
        const float ip1r = ((float *)ip1)[0] + ip2r;

        sum += ip1r * ip1r;
    }
    ((float *)op)[0] = sum;
}

NPY_NO_EXPORT void
DOUBLE_vdot_add(char *ip1, npy_intp is1, char *ip2, npy_intp is2,
    char *op, npy_intp n, void *NPY_UNUSED(ignore))
{
    double sum = (double)0.0;
    const double ip2r = ((double *)ip2)[0];
    npy_intp i;


    for (i = 0; i < n; i++, ip1 += is1) {
        const double ip1r = ((double *)ip1)[0] + ip2r;

        sum += ip1r * ip1r;
    }
    ((double *)op)[0] = sum;
}

NPY_NO_EXPORT void
LONGDOUBLE_vdot_add(char *ip1, npy_intp is1, char *ip2, npy_intp is2,
    char *op, npy_intp n, void *NPY_UNUSED(ignore))
{
    npy_longdouble sum = (npy_longdouble)0.0;
    const npy_longdouble ip2r = ((npy_longdouble *)ip2)[0];
    npy_intp i;


    for (i = 0; i < n; i++, ip1 += is1) {
        const npy_longdouble ip1r = ((npy_longdouble *)ip1)[0] + ip2r;

        sum += ip1r * ip1r;
    }
    ((npy_longdouble *)op)[0] = sum;
}

/*
 * All data is assumed aligned.
 */
NPY_NO_EXPORT void
CFLOAT_vdot_add(char *ip1, npy_intp is1, char *ip2, npy_intp is2,
    char *op, npy_intp n, void *NPY_UNUSED(ignore))
{
    float sumr = (float)0.0;
    float sumi = (float)0.0;
    const float ip2r = ((float *)ip2)[0];
    const float ip2i = ((float *)ip2)[1];
    npy_intp i;


    for (i = 0; i < n; i++, ip1 += is1) {
        const float ip1r = ((float *)ip1)[0] + ip2r;
        const float ip1i = ((float *)ip1)[1] + ip2i;

        sumr += ip1r * ip1r + ip1i * ip1i;
        sumi += ip1r * ip1i - ip1i * ip1r;
    }
    ((float *)op)[0] = sumr;
    ((float *)op)[1] = sumi;
}


/*
 * All data is assumed aligned.
 */
NPY_NO_EXPORT void
CDOUBLE_vdot_add(char *ip1, npy_intp is1, char *ip2, npy_intp is2,
             char *op, npy_intp n, void *NPY_UNUSED(ignore))
{
    double sumr = (double)0.0;
    double sumi = (double)0.0;

    const double ip2r = ((double *)ip2)[0];
    const double ip2i = ((double *)ip2)[1];
    npy_intp i;

    for (i = 0; i < n; i++, ip1 += is1) {
        const double ip1r = ((double *)ip1)[0] + ip2r;
        const double ip1i = ((double *)ip1)[1] + ip2i;

        sumr += ip1r * ip1r + ip1i * ip1i;
        sumi += ip1r * ip1i - ip1i * ip1r;
    }

    ((double *)op)[0] = sumr;
    ((double *)op)[1] = sumi;
}


/*
 * All data is assumed aligned.
 */
NPY_NO_EXPORT void
CLONGDOUBLE_vdot_add(char *ip1, npy_intp is1, char *ip2, npy_intp is2,
                 char *op, npy_intp n, void *NPY_UNUSED(ignore))
{
    npy_longdouble tmpr = 0.0L;
    npy_longdouble tmpi = 0.0L;

    const npy_longdouble ip2r = ((npy_longdouble *)ip2)[0];
    const npy_longdouble ip2i = ((npy_longdouble *)ip2)[1];
    npy_intp i;

    for (i = 0; i < n; i++, ip1 += is1) {
        const npy_longdouble ip1r = ((npy_longdouble *)ip1)[0] + ip2r;
        const npy_longdouble ip1i = ((npy_longdouble *)ip1)[1] + ip2i;

        tmpr += ip1r * ip1r + ip1i * ip1i;
        tmpi += ip1r * ip1i - ip1i * ip1r;
    }


    ((npy_longdouble *)op)[0] = tmpr;
    ((npy_longdouble *)op)[1] = tmpi;
}

/*
 * All data is assumed aligned.
NPY_NO_EXPORT void
OBJECT_vdot_add(char *ip1, npy_intp is1, char *ip2, npy_intp is2, char *op, npy_intp n,
    char *ip3, npy_intp is3,
            void *NPY_UNUSED(ignore))
{
    npy_intp i;
    PyObject *tmp0, *tmp1, *tmp2, *tmp = NULL;
    PyObject **tmp3;
    for (i = 0; i < n; i++, ip1 += is1, ip2 += is2) {
        if ((*((PyObject **)ip1) == NULL) || (*((PyObject **)ip2) == NULL)) {
            tmp1 = Py_False;
            Py_INCREF(Py_False);
        }
        else {
            tmp0 = PyObject_CallMethod(*((PyObject **)ip1), "conjugate", NULL);
            if (tmp0 == NULL) {
                Py_XDECREF(tmp);
                return;
            }
            tmp1 = PyNumber_Multiply(tmp0, *((PyObject **)ip2));
            Py_DECREF(tmp0);
            if (tmp1 == NULL) {
                Py_XDECREF(tmp);
                return;
            }
        }
        if (i == 0) {
            tmp = tmp1;
        }
        else {
            tmp2 = PyNumber_Add(tmp, tmp1);
            Py_XDECREF(tmp);
            Py_XDECREF(tmp1);
            if (tmp2 == NULL) {
                return;
            }
            tmp = tmp2;
        }
    }
    tmp3 = (PyObject**) op;
    tmp2 = *tmp3;
    *((PyObject **)op) = tmp;
    Py_XDECREF(tmp2);
}
 */
