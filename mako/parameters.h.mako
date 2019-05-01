#ifndef T
#define T ${'{0:1.3f}'.format(waveform.period)}  /* period (s) */
#endif

#ifndef M
#define M ${'{0:4d}'.format(waveform.N)} /* No. of Fourier coefs. */
#endif

#ifndef R
#define R ${'{0:.8e}'.format(waveform.radius)} /* vessel radius (m) */
#endif

#ifndef rho
#define rho ${'{0:6.1f}'.format(waveform.rho)}  /* fluid density (kg/m^3) */
#endif

#ifndef nu
#define nu ${'{0:.8e}'.format(waveform.nu)}  /* fluid visc. (m^2/s) */
#endif

#ifndef PI
#define PI 4.0*atan(1.0)
#endif

#ifndef Wo
#define Wo R*sqrt(2*PI/T/nu) /* Womersley No. */
#endif
