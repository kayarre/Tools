/* Resistance/Capacitance (two element Windkessel model) outlet boundary conditions for Fluent UDF.
   A stress free outlet is used with a time-dependent pressure.
   Pressure is assumed uniform across the outlet's faces.

   The time integration uses a semi-implicit backward Euler for the first t-step....
   and a semi-implicit second-order backward diff for all subsequent t-steps.

   i.e. this solves the ODE ---> P + RC * dP/dt = R * Q

   where:
   P is the pressure
   R is the resistance
   C is the capacitance
   Q is the volumetric flow rate

   INPUTS (enclosed in RC.h file):
   rho -- Fluid density
   Rv -- Peripherial resistance at vein outlet
   Cv -- Peripherial capacitence at vein outlet
   p_vein -- Initial guess for vein outlet pressure

   OUTPUTS:
   p -- pressure at vein outlet face

   Created by P. McGah --
*/

#include "udf.h"
#include "RC.h"

static double p, p_old, p_old_old, Q;

/* DEFINE_EXECUTE_AT_END updates pressure once per time step*/

DEFINE_EXECUTE_AT_END(pv_step)
{
  int N;

  N = N_TIME;

  /* This makes sure that the old values are non-zero if starting new simulation */
  if( (fabs(p_old) < 1e-06) && (fabs(p_old_old) < 1e-06 ) && (N > 1)){

    p_old = p;
    p_old_old = p;

  }

  /* All other t-stepts */
  p_old_old = p_old;
  p_old = p;

}

/* This applies the pressure at the outlet face */

DEFINE_PROFILE(vein_p, thread, position)
{

#if !RP_HOST

  face_t f;

  int count, N;

  real R, C, dt;

  R = (double)R_${bc_name};
  C = (double)C_${bc_name};

  dt = CURRENT_TIMESTEP;

  N = N_TIME;

  /* F_FLUX -- Built in function to get mass flow rate at outlet  */
  /* Divides by rho to get volumetric rate  */

  if(N_TIME == 1){

    p = p_${bc_name};
    p_old = p_${bc_name};
    p_old_old = p_${bc_name};

  }

  if( (fabs(p) < 1e-06) && (N_TIME > 1) )
    {
      count = 0;
      begin_f_loop(f, thread)
	{

	  p += F_P(f, thread);

	  count += 1;

	}
      end_f_loop(f, thread)

	p = p / (double)count;
      p_old = p;
      p_old_old = p;

    }

  /***  Calculate Flow Rate at t-step n+1  ****/
  Q = 0.;
  begin_f_loop(f, thread)
    {

      Q += F_FLUX(f, thread) / rho;

    }
  end_f_loop(f, thread)

    /*** Update Face pressure  ****/

    if(N == 1){

      p = (R*Q + (R*C)/dt * p_old) / (1. + (R*C)/dt);

    }
    else{

      p = (R*Q + 2.*(R*C)/dt * p_old - 0.5*(R*C)/dt * p_old_old) / (1. + 1.5*(R*C)/dt);

    }

    begin_f_loop(f, thread)
    {

      F_PROFILE(f, thread, position) = p;

    }
  end_f_loop(f, thread)

#endif

}
