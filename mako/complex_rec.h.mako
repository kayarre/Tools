/* C Library: complex number data type declaration
   Taken from Numerical Recipies in C */

typedef struct DCOMPLEX {double r,i;} dcomplex;
  
  double Cabs(dcomplex z);
  dcomplex besselJ(int n, dcomplex y);
  dcomplex Cadd(dcomplex a, dcomplex b);
  dcomplex Csub(dcomplex a, dcomplex b);
  dcomplex Cmul(dcomplex a, dcomplex b);
  dcomplex Complex(double re, double im);
  dcomplex Cdiv(dcomplex a, dcomplex b);
  dcomplex RCmul(double x, dcomplex a);
