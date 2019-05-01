/* C library which difines complex number arithmetic operations.
   Taken from Numerical Recipies in C
   Written by P. McGah */

dcomplex Cadd(a,b)
     dcomplex a,b;
{
  dcomplex c;
  c.r=a.r+b.r;
  c.i=a.i+b.i;
  return c;
}

dcomplex Csub(a,b)
     dcomplex a,b;
{
  dcomplex c;
  c.r=a.r-b.r;
  c.i=a.i-b.i;
  return c;
}

dcomplex Cmul(a,b)
     dcomplex a,b;
{
  dcomplex c;
  c.r=a.r*b.r-a.i*b.i;
  c.i=a.i*b.r+a.r*b.i;
  return c;
}

dcomplex Complex(re,im)
     float re,im;
{
  dcomplex c;
  c.r=re;
  c.i=im;
  return c;
}

dcomplex Cdiv(a,b)
     dcomplex a,b;
{ 
  dcomplex c;
  float r, den;
  if (fabs(b.r)>=fabs(b.i)){
    r=b.i/b.r;
    den=b.r+r*b.i;
    c.r=(a.r+r*a.i)/den;
    c.i=(a.i-r*a.r)/den;
  }else {
    r=b.r/b.i;
    den=b.i+r*b.r;
    c.r=(a.r*r+a.i)/den;
    c.i=(a.i*r-a.r)/den;
  }
  
  return c;
}

double Cabs(z)
     dcomplex z;
{
  double x,y,ans,temp;
  x=fabs(z.r);
  y=fabs(z.i);
  if (x == 0.0)
    ans=y;
  else if (y == 0.0)
    ans=x;
  else if (x > y){
    temp=x/y;
    ans=x*sqrt(1.0+temp*temp);
  }else{ 
    temp=x/y;
    ans=y*sqrt(1.0+temp*temp);
  } 
  return ans;
}

dcomplex RCmul(x,a)
     float x;
     dcomplex a;
{
  dcomplex c;
  c.r=x*a.r;
  c.i=x*a.i;
  return c;
}

  



  

