/* C Library: Computes Bessel functions of 1st kind integar order
   with a complex argument using the series representation.
   Taken from Num. Rec. in C
   Written by P. McGah

 */

dcomplex besselJ(int n,dcomplex y)
{
  dcomplex z, zarg, zbes;
  int i;

  zarg=RCmul(-0.25,Cmul(y,y));
  z=Complex(1.0,0.0);
  zbes=Complex(1.0,0.0);
  i=1;
  while(Cabs(z)>1e-15 && i<=10000){
    z=Cmul(z,RCmul(1.0/i/(i+n),zarg));
    if(Cabs(z)<=1e-15) break;
    zbes=Cadd(zbes,z);
    i++;
  }
  zarg=RCmul(0.5,y);
  for(i=1;i<=n;i++){
    zbes=Cmul(zbes,zarg);
  }
  return zbes;
}
