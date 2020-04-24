/***************************************************************/

/* Womersley inlet velocity profile with Bessel Functions for Fluent */
/* Fourier Amplitudes are computed in a companion Matlab file "Wom_wave.m" */

/* Libraries info:

   udf.h -- Fluent UDF library
   complex_rec.h -- complex data type declaration
   zbes.h -- computes Bessel function of 1st kind integer order; series representation
   complex_ops.h -- complex arithmetic operations declaration
   parameters.h -- fluid parameters (viscosity, density, etc....)

*/

/* Created by P. McGah */
/**************************************************************/

#include "udf.h"
#include "complex_rec.h"
#include "zbes.h"
#include "complex_ops.h"
#include "parameters.h"

FILE *fourier;
double vel_i[M], vel_r[M];
double w_vel(double alpha, int nfour, double *vel_r, double *vel_i, double r, double t);


/* Main Program -- Defines velocity at inlet zone */

DEFINE_PROFILE(prox_art_vel, thread, position)
{
  int n;

  double t, r;
  real x[ND_ND];

  double xo = ${'{0:.8e}'.format(bc["X"])}; /* meters */
  double yo = ${'{0:.8e}'.format(bc["Y"])};
  double zo = ${'{0:.8e}'.format(bc["Z"])};

  face_t f;

  /**************************************/
  /* LOADS FOURIER AMPLITUDES */

  fourier=fopen("${coef_file_name}","r");

  for (n=0;n<M;n++){
    fscanf(fourier,"%lf", &vel_i[n]);  /* Sine comp. */
  }
  for (n=0;n<M;n++){
    fscanf(fourier,"%lf", &vel_r[n]);  /* Cosine comp. */
  }
  fclose(fourier);

  /*************************************/

  t = RP_Get_Real("flow-time");

  begin_f_loop(f, thread)
    {
      F_CENTROID(x,f,thread);
      r = sqrt( (x[2]-zo)*(x[2]-zo) + (x[1]-yo)*(x[1]-yo) + (x[0]-xo)*(x[0]-xo) )/R;

      F_PROFILE(f, thread, position) = w_vel(Wo, M, vel_r, vel_i, r, t);
    }
  end_f_loop(f, thread);

}

/**************************************************************/

/* Function to compute Bessel Amplitudes of Womersley profile */

double w_vel(double alpha, int nfour, double *vel_r, double *vel_i, double r, double t){
  dcomplex zi, z1;
  double w;
  double kt;
  int k;
  dcomplex za, zar, zJ0, zJ0r, zq, zvel, zJ0rJ0;

  zi = Complex(0.0,1.0);
  z1 = Complex(1.0,0.0);
  w = vel_r[0]*(1-r*r); /* Poiseulle Flow Component */
  for (k=1;k<nfour;k++){
    kt = 2*PI*k*t/T;
    za = RCmul(alpha*sqrt(k)/sqrt(2),Complex(-1.0,1.0));
    zar = RCmul(r,za);
    zJ0 = besselJ(0,za);
    zJ0r = besselJ(0,zar);
    zJ0rJ0 = Cdiv(zJ0r,zJ0);
    zq = Cmul(Complex(vel_r[k], vel_i[k]),Complex(cos(kt), sin(kt)));
    zvel = Cmul(zq,Csub(z1,zJ0rJ0));
    w = w+zvel.r;
  }

  return w;
}
