/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) jit_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s1[23] = {4, 4, 0, 4, 8, 12, 16, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};

/* T_fk:(i0[6])->(o0[4x4]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a4, a5, a6, a7, a8, a9;
  a0=-7.0710678182113929e-01;
  a1=arg[0]? arg[0][0] : 0;
  a2=cos(a1);
  a3=1.7948965149208059e-09;
  a4=arg[0]? arg[0][1] : 0;
  a5=cos(a4);
  a6=(a3*a5);
  a4=sin(a4);
  a6=(a6-a4);
  a7=(a2*a6);
  a8=arg[0]? arg[0][2] : 0;
  a9=cos(a8);
  a10=(a7*a9);
  a11=(a3*a4);
  a11=(a11+a5);
  a12=(a2*a11);
  a8=sin(a8);
  a13=(a12*a8);
  a10=(a10-a13);
  a13=arg[0]? arg[0][3] : 0;
  a14=cos(a13);
  a15=(a3*a14);
  a13=sin(a13);
  a15=(a15-a13);
  a16=(a10*a15);
  a7=(a7*a8);
  a17=(a12*a9);
  a7=(a7+a17);
  a17=(a3*a13);
  a17=(a14+a17);
  a18=(a7*a17);
  a16=(a16-a18);
  a18=arg[0]? arg[0][4] : 0;
  a19=cos(a18);
  a20=(a16*a19);
  a1=sin(a1);
  a18=sin(a18);
  a21=(a1*a18);
  a20=(a20-a21);
  a21=arg[0]? arg[0][5] : 0;
  a22=cos(a21);
  a23=(a20*a22);
  a24=(a3*a13);
  a24=(a24+a14);
  a10=(a10*a24);
  a14=(a3*a14);
  a14=(a14-a13);
  a13=(a7*a14);
  a10=(a10+a13);
  a21=sin(a21);
  a13=(a10*a21);
  a23=(a23-a13);
  a13=(a0*a23);
  a25=-7.0710678055195575e-01;
  a16=(a16*a18);
  a26=(a1*a19);
  a16=(a16+a26);
  a26=(a3*a16);
  a20=(a20*a21);
  a27=(a10*a22);
  a20=(a20+a27);
  a26=(a26+a20);
  a27=(a25*a26);
  a13=(a13-a27);
  a27=-3.5897930298416118e-09;
  a20=(a3*a20);
  a20=(a20-a16);
  a28=(a27*a20);
  a13=(a13+a28);
  if (res[0]!=0) res[0][0]=a13;
  a6=(a1*a6);
  a13=(a6*a9);
  a11=(a1*a11);
  a28=(a11*a8);
  a13=(a13-a28);
  a28=(a13*a15);
  a6=(a6*a8);
  a29=(a11*a9);
  a6=(a6+a29);
  a29=(a6*a17);
  a28=(a28-a29);
  a29=(a28*a19);
  a30=(a2*a18);
  a29=(a29+a30);
  a30=(a29*a22);
  a13=(a13*a24);
  a31=(a6*a14);
  a13=(a13+a31);
  a31=(a13*a21);
  a30=(a30-a31);
  a31=(a0*a30);
  a32=(a2*a19);
  a28=(a28*a18);
  a32=(a32-a28);
  a28=(a3*a32);
  a29=(a29*a21);
  a33=(a13*a22);
  a29=(a29+a33);
  a28=(a28-a29);
  a33=(a25*a28);
  a31=(a31+a33);
  a29=(a3*a29);
  a29=(a32+a29);
  a33=(a27*a29);
  a31=(a31+a33);
  if (res[0]!=0) res[0][1]=a31;
  a31=(a3*a4);
  a31=(a5+a31);
  a33=(a31*a9);
  a5=(a3*a5);
  a5=(a5-a4);
  a4=(a5*a8);
  a33=(a33+a4);
  a15=(a33*a15);
  a9=(a5*a9);
  a31=(a31*a8);
  a9=(a9-a31);
  a17=(a9*a17);
  a15=(a15+a17);
  a18=(a15*a18);
  a17=(a3*a18);
  a14=(a9*a14);
  a33=(a33*a24);
  a14=(a14-a33);
  a33=(a14*a22);
  a15=(a15*a19);
  a19=(a15*a21);
  a33=(a33-a19);
  a17=(a17-a33);
  a19=(a25*a17);
  a15=(a15*a22);
  a21=(a14*a21);
  a15=(a15+a21);
  a0=(a0*a15);
  a19=(a19-a0);
  a3=(a3*a33);
  a3=(a18+a3);
  a27=(a27*a3);
  a19=(a19+a27);
  if (res[0]!=0) res[0][2]=a19;
  a19=0.;
  if (res[0]!=0) res[0][3]=a19;
  a27=(a25*a23);
  a33=7.0710678182113929e-01;
  a0=(a33*a26);
  a27=(a27-a0);
  if (res[0]!=0) res[0][4]=a27;
  a0=(a25*a30);
  a21=(a33*a28);
  a0=(a0+a21);
  if (res[0]!=0) res[0][5]=a0;
  a33=(a33*a17);
  a25=(a25*a15);
  a33=(a33-a25);
  if (res[0]!=0) res[0][6]=a33;
  if (res[0]!=0) res[0][7]=a19;
  a25=2.5383669967352592e-09;
  a23=(a25*a23);
  a21=2.5383669921791530e-09;
  a26=(a21*a26);
  a23=(a23-a26);
  a23=(a23-a20);
  if (res[0]!=0) res[0][8]=a23;
  a30=(a25*a30);
  a28=(a21*a28);
  a30=(a30+a28);
  a30=(a30-a29);
  if (res[0]!=0) res[0][9]=a30;
  a21=(a21*a17);
  a25=(a25*a15);
  a21=(a21-a25);
  a21=(a21-a3);
  if (res[0]!=0) res[0][10]=a21;
  if (res[0]!=0) res[0][11]=a19;
  a19=-9.5000000000000001e-02;
  a27=(a19*a27);
  a21=1.3999999999999999e-02;
  a25=(a21*a20);
  a15=1.1200000000000000e-02;
  a17=(a15*a20);
  a30=4.0299999999999996e-02;
  a20=(a30*a20);
  a28=1.1570000000000000e-01;
  a10=(a28*a10);
  a23=5.7230000000000003e-01;
  a7=(a23*a7);
  a26=6.1199999999999999e-01;
  a12=(a26*a12);
  a22=-1.7190000000000000e-01;
  a24=(a22*a1);
  a12=(a12-a24);
  a24=2.2094100000000000e-01;
  a31=(a24*a1);
  a12=(a12-a31);
  a7=(a7+a12);
  a12=1.1490000000000000e-01;
  a1=(a12*a1);
  a7=(a7-a1);
  a10=(a10+a7);
  a7=9.2200000000000004e-02;
  a16=(a7*a16);
  a10=(a10-a16);
  a20=(a20+a10);
  a17=(a17+a20);
  a25=(a25+a17);
  a27=(a27+a25);
  if (res[0]!=0) res[0][12]=a27;
  a0=(a19*a0);
  a27=(a21*a29);
  a25=(a15*a29);
  a29=(a30*a29);
  a32=(a7*a32);
  a13=(a28*a13);
  a12=(a12*a2);
  a6=(a23*a6);
  a22=(a22*a2);
  a11=(a26*a11);
  a22=(a22+a11);
  a24=(a24*a2);
  a22=(a22+a24);
  a6=(a6+a22);
  a12=(a12+a6);
  a13=(a13+a12);
  a32=(a32+a13);
  a29=(a29+a32);
  a25=(a25+a29);
  a27=(a27+a25);
  a0=(a0+a27);
  if (res[0]!=0) res[0][13]=a0;
  a19=(a19*a33);
  a21=(a21*a3);
  a15=(a15*a3);
  a30=(a30*a3);
  a7=(a7*a18);
  a28=(a28*a14);
  a23=(a23*a9);
  a26=(a26*a5);
  a5=1.2730000000000000e-01;
  a26=(a26+a5);
  a23=(a23+a26);
  a28=(a28+a23);
  a7=(a7+a28);
  a30=(a30+a7);
  a15=(a15+a30);
  a21=(a21+a15);
  a19=(a19+a21);
  if (res[0]!=0) res[0][14]=a19;
  a19=1.;
  if (res[0]!=0) res[0][15]=a19;
  return 0;
}

CASADI_SYMBOL_EXPORT int T_fk(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int T_fk_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int T_fk_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void T_fk_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int T_fk_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void T_fk_release(int mem) {
}

CASADI_SYMBOL_EXPORT void T_fk_incref(void) {
}

CASADI_SYMBOL_EXPORT void T_fk_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int T_fk_n_in(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_int T_fk_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real T_fk_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* T_fk_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* T_fk_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* T_fk_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* T_fk_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int T_fk_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
