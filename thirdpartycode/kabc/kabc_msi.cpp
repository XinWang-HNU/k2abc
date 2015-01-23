// prior X1 is THETA(mutation rate)~Gamma(100*th0,0.01)
// prior X2 is TMRCA(time to most recent common ancestor)=Tn+...+T2, Ti~Exp(i(i-1)/2)
// cond. val. Y is segregating site~Poisson(X1*length*th/2), length=nTn+...+2T2

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define THETA0 5.521656 
#define TMRCA0 1.856026
// true posterior obtained by 10^8 acceptance by rejection for 
// num. of seq. 63 seg. sites 26

#define LOOP 100
#define TRIAL 100 // simulated sample size

#define NMAX 20 //20    // maximum rank of low rank approx.
#define TOL 0.1 //0.1    // tolerance for low rank approx.

#define EPS (0.1/TRIAL) // regularization paramater (to be chosen)

#define PI 3.1415926536
#define TINY 1.0e-20;

int main(int argc,char **argv) {

  using namespace std;

  void init_by_array(unsigned long init_key[], int key_length);
  unsigned long init[4]={0x123, 0x234, 0x345, 0x456}, length=4;
  // for initialize Mersenne Twistor

  double genrand_real2(void); // uniform deviate
  int poidev(double xm);      // poisson deviate
  double expdev(double lam);  // exponential deviate
  double gamdev(double a);    // gamma deviate

  double kerY(double y1,double y2,double sig2); // kernel func.

  void LRAKY(double KY[],int pp[],double y[],double sig2,int *lr);
  // low rank approx.
  int rank;

  int ns=atoi(argv[1]);  // size of observed sample
  int seg=atoi(argv[2]); // num. of observed segregating sites

  double r;
  double tmp;
  int i,j,k,l;

  double th0;         // Watterson's Estimator
  double th;          // mutation rate
  double le;          // length of genealogy
  double tm;          // TMRCA

  double meanX1,varX1; // post. mean and var of X1
  double meanX2,varX2; // post. mean and var of X2
  double avmeX1;
  double avmeX2;       // average over loops
  double mse;          // mean squared error

  double meanX1a,varX1a; // post. mean and var of X1
  double meanX2a,varX2a; // post. mean and var of X2
  double avmeX1a;
  double avmeX2a;       // average over loops
  double msea;          // mean squared error 
  // low rank approx.

  double X1[TRIAL]; // param. 1
  double X2[TRIAL]; // param. 2
  double Y[TRIAL];  // target

  int pp[TRIAL];   // permutation matrix for low rank approx.
  double *KY,*RY; // K*KT in low rank approx.

  double *gy; // Gram matrix

  double *mtmp,*mtmp2;
  double vtmp[TRIAL],vtmp2[TRIAL];

  KY=(double*)calloc(TRIAL*NMAX,sizeof(double));
  RY=(double*)calloc(TRIAL*NMAX,sizeof(double));
  gy=(double*)calloc(TRIAL*TRIAL,sizeof(double));
  mtmp=(double*)calloc(TRIAL*TRIAL,sizeof(double));
  mtmp2=(double*)calloc(TRIAL*TRIAL,sizeof(double));

  double mu[TRIAL],nu[TRIAL];
  double ky[TRIAL];

  double sig2;   // band width;
  double *sig2s;
  sig2s=(double*)calloc(TRIAL,sizeof(double));

  int cmp(const void *a,const void *b); // for qsort

  void inverse(double *m,int tn); // for matrix inversion

  init_by_array(init, length); // initialize Mersenne Twistor

  cout << "observed sample size " << ns << endl;
  cout << "observed segregating sites " << seg << endl; 
  cout << "simulated sample size " << TRIAL << endl;

  th0=1.0;
  for(i=2;i<=ns-1;i++) th0+=1.0/i;
  th0=seg/th0;
  cout << "Watterson's Estimator of mutation rate " << th0 << endl;
  
  avmeX1=0.0;
  avmeX2=0.0;
  mse=0.0;
  
  avmeX1a=0.0;
  avmeX2a=0.0;
  msea=0.0;
  
  for(k=0;k<LOOP;k++) {
    
    for(i=0;i<TRIAL;i++) {
      tm=0.0;
      le=0.0;
      for(j=2;j<=ns;j++) {
        r=expdev(j*(j-1)*0.5);
        le+=j*r;
        tm+=r;
      }
      th=0.01*gamdev(100.0*th0); // ~Gamma(100*th0,0.01)
      X1[i]=th;
      X2[i]=tm;
      Y[i]=(double)poidev(le*th*0.5);
    } // simulation, to be obtained as an input file

    for(i=0;i<TRIAL-1;i++) {
      sig2s[i]=(Y[i+1]-Y[i])*(Y[i+1]-Y[i]);
    }
    qsort(sig2s,TRIAL-1,sizeof(double),(int(*)(const void*,const void*))cmp);
    sig2=sig2s[(TRIAL-1)/2];
    // fixing band width by median, to be chosen by cross varidation
    
    for(i=0;i<TRIAL;i++) ky[i]=kerY(Y[i],seg,sig2);
    
    LRAKY(KY,pp,Y,sig2,&rank);
    
    for(i=0;i<TRIAL;i++) {
      for(j=0;j<rank;j++) {
	RY[rank*pp[i]+j]=KY[i+TRIAL*j];
      }
    }
    
    // low rank approx.

    /* 
    for(i=0;i<TRIAL;i++) {
      for(j=0;j<TRIAL;j++) {
	gy[TRIAL*i+j]=kerY(Y[i],Y[j],sig2); 
      }
    } // exact
    */

    tmp=1.0/TRIAL/EPS;
    for(i=0;i<rank;i++) {
      for(j=0;j<rank;j++) {
        if(i==j) mtmp[rank*i+j]=TRIAL*EPS*1.0;
        else mtmp[rank*i+j]=0.0;
        for(l=0;l<TRIAL;l++) {
          mtmp[rank*i+j]+=RY[rank*l+i]*RY[rank*l+j];
        }
      }
    }
    inverse(mtmp,rank);    
    for(i=0;i<rank;i++) {
      vtmp[i]=0.0;
      for(j=0;j<TRIAL;j++)
        vtmp[i]+=RY[rank*j+i]*ky[j];
    }
    for(i=0;i<rank;i++) {
      vtmp2[i]=0.0;
      for(j=0;j<rank;j++)
        vtmp2[i]+=mtmp[rank*i+j]*vtmp[j];
    }
    for(i=0;i<TRIAL;i++) {
      vtmp[i]=0.0;
      for(j=0;j<rank;j++)
        vtmp[i]+=RY[rank*i+j]*vtmp2[j];
    }
    for(i=0;i<TRIAL;i++) {
      mu[i]=tmp*(ky[i]-vtmp[i]);
    }  // low rank approx.
    //mu=tmp*(in-RY*(RY.transpose()*RY+TRIAL*EPS*ir).inverse()*RY.transpose())*ky; // in octave

    /*    
    for(i=0;i<TRIAL;i++) {
      for(j=0;j<TRIAL;j++) {
        mtmp[TRIAL*i+j]=gy[TRIAL*i+j];
        if(i==j) mtmp[TRIAL*i+j]+=TRIAL*EPS*1.0;
      }
    }
    inverse(mtmp,TRIAL);
    for(i=0;i<TRIAL;i++) {
      nu[i]=0.0;
      for(j=0;j<TRIAL;j++) {
        nu[i]+=mtmp[TRIAL*i+j]*ky[j];
      }
    }
    */ // exact

    meanX1=varX1=0.0;
    meanX1a=varX1a=0.0;
    for(i=0;i<TRIAL;i++) {
      //meanX1+=nu[i]*X1[i];
      //varX1+=nu[i]*X1[i]*X1[i];
      meanX1a+=mu[i]*X1[i];
      varX1a+=mu[i]*X1[i]*X1[i];
    } 
    varX1-=meanX1*meanX1;
    varX1a-=meanX1a*meanX1a;

    meanX2=varX2=0.0;
    meanX2a=varX2a=0.0;
    for(i=0;i<TRIAL;i++) {
      //meanX2+=nu[i]*X2[i];
      //varX2+=nu[i]*X2[i]*X2[i];
      meanX2a+=mu[i]*X2[i];
      varX2a+=mu[i]*X2[i]*X2[i];
    } 
    varX2-=meanX2*meanX2;
    varX2a-=meanX2a*meanX2a;

    cout << "THETA mean " << meanX1 << " var " << varX1;
    cout << " Approx. THETA mean " << meanX1a << " var " << varX1a << endl;
 
    cout << "TMRCA mean " << meanX2 << " var " << varX2;
    cout << " Approx. TMRCA mean " << meanX2a << " var " << varX2a << endl;
  
    avmeX1+=meanX1/LOOP;
    avmeX2+=meanX2/LOOP;
    avmeX1a+=meanX1a/LOOP;
    avmeX2a+=meanX2a/LOOP;

    mse+=(meanX1-THETA0)*(meanX1-THETA0)/LOOP+(meanX2-TMRCA0)*(meanX2-TMRCA0)/LOOP;
    msea+=(meanX1a-THETA0)*(meanX1a-THETA0)/LOOP+(meanX2a-TMRCA0)*(meanX2a-TMRCA0)/LOOP;
    
  }    
  
  cout << "THETA " << avmeX1 << " TMRCA " << avmeX2 << " MSE " << mse;
  cout << " APPROX. THETA " << avmeX1a << " TMRCA " << avmeX2a << " MSE ";
  cout << msea << endl;

  free(KY);
  free(gy);
  free(mtmp);
  free(mtmp2);
  free(sig2s);

  return 0;
}

double expdev(double lam) {
  
  double genrand_real2(void);
  
  return -log(genrand_real2())/lam;
}

int cmp(const void *a,const void *b) {
     return (*(double *)a>*(double *)b)? 1: -1;
}

int poidev(double xm)
{
  double gammln(double xx);
  double genrand_real2(void);
 
  static double sq,alxm,g,oldm=(-1.0);
  double em,t,y;
  
  if (xm < 12.0) {
    if (xm != oldm) {
      oldm=xm;
      g=exp(-xm);
    }
    em = -1;
    t=1.0;
    do {
      em += 1.0;
      t *= genrand_real2();
    } while (t > g);
  } 
  else {
    if (xm != oldm) {
      oldm=xm;
      sq=sqrt(2.0*xm);
      alxm=log(xm);
      g=xm*alxm-gammln(xm+1.0);
    }
    do {
      do {
	y=tan(PI*(genrand_real2()));
	em=sq*y+xm;
      } while (em < 0.0);
      em=floor(em);
      t=0.9*(1.0+y*y)*exp(em*alxm-gammln(em+1.0)-g);
    } while (genrand_real2() > t);
  }
  return (int)em;
}

double gammln(double a)
{
  int i;
  double b,tmp;
  static double coeff[6]={76.18009172947146,-86.50532032941677,24.01409824083091,-1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5};

  b=a;
  tmp=1.000000000190015;
  for(i=0;i<6;i++) tmp+=coeff[i]/++a;

  return 0.918938533204673-(b+5.5)+(b+0.5)*log(b+5.5)+log(tmp/b);
}

double kerY(double y1,double y2,double sig2) {
  return exp(-0.5*(y1-y2)*(y1-y2)/sig2);
}

static double randev1(double a)
{
  //gamma dev.: a>=1
  //GBEST algorithm  (D.J. BEST: Appl. Stat. 29 p 181 1978)

  double genrand_real2();

  double x, d, e, c, g, f, r1, r2;

  e = a - 1.0;
  c = 3.0 * a - 0.75;

  for (;;) {
    r1 = genrand_real2();
    g = r1 - (r1 * r1);
    if (g <= 0.0) continue;
    f = (r1 - 0.5) * sqrt(c / g);
    x = e + f;
    if (x <= 0.0) continue;
    r2 = genrand_real2();
    d = 64.0 * r2 * r2 * g * g * g;
    if ((d >= 1.0 - 2.0 * f * f / x) && (log(d) >= 2.0 * (e * log(x / e) - f))) continue;
    return (x);
  }
}

static double randev0(double a)
{
  // Gamma dev.: a < 1

  double genrand_real2();

  double r1, r2, x, w;
  double t = 1.0 - a;
  double p = t / (t + a * exp(-t));
  double s = 1.0 / a;
  for (;;) {
    r1 = genrand_real2();
    if (r1 <= p) {
      x = t * pow(r1 / p, s);
      w = x;
    } else {
      x = t + log((1.0 - p) / (1.0 - r1));
      w = t * log(x / t);
    }
    r2 = genrand_real2();
    if ((1.0 - r2) <= w) {
      if ((1.0 / r2 - 1.0) <= w) continue;
    }
    if (x==0.0) x = 1.0e-20;
    return x;
  }

}

double gamdev(double a)
{

  double randev0(double a);
  double randev1(double a);
  double expdev(double a); 

  if (a < 1.0) {
    return( randev0(a));
  } if (a == 1.0) {
    return(expdev(1.0));
  }
    return(randev1(a));
}

void LRAKY(double G[],int pp[],double y[],double sig,int *lr) {

     int jast,i,j,iter;
     int n=TRIAL;
     double a,b,residual,maxdiagG;
     double *diagG;

     diagG=(double*)calloc(n,sizeof(double));
     
     iter=0;
     residual=TRIAL;

     for(i=0;i<=n-1;i++) pp[i]=i;
     for(i=0;i<=n-1;i++) diagG[i]=1;
	  
     jast=0;

     while(residual>TOL&&iter<NMAX) {
     
	  if(jast!=iter) {
	       i=pp[jast]; pp[jast]=pp[iter]; pp[iter]=i;
	       for(i=0;i<=iter;i++) {
		  a=G[jast+n*i];G[jast+n*i]=G[iter+n*i];
                  G[iter+n*i]=a;
     	       }
	  }
	  G[iter*(n+1)]=sqrt(diagG[jast]);
	  for(i=iter+1;i<=n-1;i++) {
	       G[i+n*iter]=exp(-0.5/sig*(pow(y[pp[iter]]-y[pp[i]],2.0)));
	  }
	  if(iter>0)
	       for(j=0;j<=iter-1;j++)
		    for(i=iter+1;i<=n-1;i++) G[i+n*iter]-=G[i+n*j]*G[iter+n*j];
	  for(i=iter+1;i<=n-1;i++) G[i+n*iter]/=G[iter*(n+1)];
	  residual=0.0;
	  jast=iter+1;
	  maxdiagG=0;
	  for(i=iter+1;i<=n-1;i++) {
              b=1.0;
	       for(j=0;j<=iter;j++)
		    b-=G[i+j*n]*G[i+n*j];
	       diagG[i]=b;
	       if(b>maxdiagG) {
		    jast=i;
		    maxdiagG=b;
	       }
	       residual+=b;
	  }

	  iter++;
	  	  
     }
     //std::cout << "KY" << iter << " " << residual << std::endl;
     *lr=iter;     

     free(diagG);
     
     return;

}  

void inverse(double *m,int tn) {
  double *y,d,*col;
  int i,j,*indx;
  
  void lubksb(double *a, int n, int *indx, double b[]);
  void ludcmp(double *a, int n, int *indx, double *d);

  indx=(int*)calloc(tn,sizeof(int));  
  col=(double*)calloc(tn,sizeof(double));  
  y=(double*)calloc(tn*tn,sizeof(double));  

  ludcmp(m,tn,indx,&d);
  
  for(j=0;j<tn;j++) {
    for(i=0;i<tn;i++) col[i]=0.0;
    col[j]=1.0;
    lubksb(m,tn,indx,col);
    for(i=0;i<tn;i++) y[tn*i+j]=col[i];
  }
  
  for(i=0;i<tn*tn;i++) m[i]=y[i];

  free(indx);
  free(col);
  free(y);  

  return;
}

void ludcmp(double *a, int n, int *indx, double *d)
{
  int i,imax,j,k;
  double big,dum,sum,temp;
  double *vv;
  
  vv=(double*)calloc(n,sizeof(double));
  
  *d=1.0;
  for (i=0;i<n;i++) {
    big=0.0;
    for (j=0;j<n;j++)
      if ((temp=fabs(a[n*i+j])) > big) big=temp;
    if (big == 0.0) { 
      std::cout << "singular matrix in routine ludcmp" << std::endl;
    }
    vv[i]=1.0/big;
  }
  
  for (j=0;j<n;j++) {
    
    for (i=0;i<j;i++) {
      sum=a[n*i+j];
      for (k=0;k<i;k++) sum -= a[n*i+k]*a[n*k+j];
      a[n*i+j]=sum;
    }
    
    big=0.0;
    for (i=j;i<n;i++) {
      sum=a[n*i+j];
      for (k=0;k<j;k++)
	sum -= a[n*i+k]*a[n*k+j];
      a[n*i+j]=sum;
      if ( (dum=vv[i]*fabs(sum)) >= big) {
	big=dum;
	imax=i;
      }
    }

    if (j != imax) {
      for (k=0;k<n;k++) {
	dum=a[n*imax+k];
	a[n*imax+k]=a[n*j+k];
	a[n*j+k]=dum;
      }
      *d = -(*d);
      vv[imax]=vv[j];
    }
    
    indx[j]=imax;
    if (a[n*j+j] == 0.0) a[n*j+j]=TINY;
    
    if (j != n) {
      dum=1.0/(a[n*j+j]);
      for (i=j+1;i<n;i++) a[n*i+j] *= dum;
    }
    
  }
  
  free(vv);
}

void lubksb(double *a, int n, int *indx, double b[])
{
  int i,ii=0,ip,j;
  double sum;

  for (i=0;i<n;i++) {
    ip=indx[i];
    sum=b[ip];
    b[ip]=b[i];
    if (ii)
      for (j=ii-1;j<=i-1;j++) sum -= a[n*i+j]*b[j];
    else if (sum) ii=i+1;
    b[i]=sum;
  }
  for (i=n-1;i>=0;i--) {
    sum=b[i];
    for (j=i+1;j<n;j++) sum -= a[n*i+j]*b[j];
    b[i]=sum/a[n*i+i];
  }
}
