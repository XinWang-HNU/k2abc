// prior X1 is THETA(mutation rate)~Gamma(th0,1), th0 is Watterson's Estimator
// prior X1 is THETA(mutation rate)~Gamma(100*th0,0.01)
// prior X2 is TMRCA(time to most recent common ancestor)=Tn+...+T2, Ti~Exp(i(i-1)/2)
// cond. val. Y is segregating site~Poisson(X1*length*th/2), length=nTn+...+2T2

#include <iostream>
#include <math.h>
#include <octave/config.h>
#include <octave/Matrix.h>

#define THETA0 5.521656
#define TMRCA0 1.856026
// obtained by rejection for num. of sequences 63 segregating sites 26 
//#define THETA0 5.521656 //Gamma(100*th0,0.01)                                //#define TMRCA0 1.856026   

#define LOOP 100
#define TRIAL 5000 // simulated sample size
#define NMAX TRIAL // TRIAL   // maximum rank of low rank approx.
#define TOL 0.0 //0.0   // tolerance for low rank approx. 

#define EPS (1.0/TRIAL) // reg. param. for mu
#define DEL 2.0*EPS   // reg. param. for we

#define PI 3.1415926536

int main(int argc,char **argv) {

  using namespace std;

  void init_by_array(unsigned long init_key[], int key_length);
  unsigned long init[4]={0x123, 0x234, 0x345, 0x456}, length=4;
  // for initialize Mersenne Twistor

  double genrand_real2(void); // uniform dev.
  int poidev(double xm);      // poisson dev.
  double expdev(double lam);  // exponential dev.
  double gamdev(double a);    // gamma dev.

  double kerX(double xa1,double xb1,double xa2,double xb2,double sig2);  
  double kerY(double y1,double y2,double sig2);  
  // kernel func.

  void LRAKX(long double KX[],int pp[],double x1[],double x2[],double sig2,int *lr);
  void LRAKY(long double KY[],int pp[],double y[],double sig2,int *lr);
  // low rank approx.

  int ns=atoi(argv[1]);  // size of observed sample
  int seg=atoi(argv[2]); // num. of observed segregating sites

  double r;
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

  double X1[TRIAL];
  double X2[TRIAL];
  double Y[TRIAL];

  ColumnVector ky(TRIAL);
  Matrix lam(TRIAL,TRIAL);
  double mupia0[TRIAL];
  ColumnVector mupia(TRIAL); // ^mu_pi with approx.
  ColumnVector mua0(TRIAL);
  ColumnVector mua(TRIAL);   // ^mu with approx.
  ColumnVector wea0(TRIAL);
  ColumnVector wea(TRIAL);   // weight with approx.

  int pp[TRIAL];         // permutation matrix for low rank approx.
  long double *KX,*KY;   // K*KT=kernel in low rank approx.
  int rankXv; int *rankX; rankX=&rankXv; 
  int rankYv; int *rankY; rankY=&rankYv;
  // ranks in low rank approx.  
  
  KX=(long double*)calloc(TRIAL*NMAX,sizeof(long double));
  KY=(long double*)calloc(TRIAL*NMAX,sizeof(long double));

  double sig2X,sig2Y;
  double *sig2Xs,*sig2Ys;
  sig2Xs=(double*)calloc(TRIAL,sizeof(double));
  sig2Ys=(double*)calloc(TRIAL,sizeof(double));
  int cmp(const void *a,const void *b); // for qsort
  //for fixing sigma sq. by median

  
  ColumnVector mupi(TRIAL);
  Matrix gx(TRIAL,TRIAL);
  Matrix gy(TRIAL,TRIAL);
  //for comparison with exact comp.
  

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

  for(k=0;k<LOOP;k++) {

    for(i=0;i<TRIAL;i++) {
      tm=0.0;
      le=0.0;
      for(j=2;j<=ns;j++) {
        r=expdev(j*(j-1)*0.5);
        le+=j*r;
        tm+=r;
      }
      //th=gamdev(th0); // ~Gamma(th0,1)
      th=0.01*gamdev(th0*100.0); // ~Gamma(100*th0,0.01)

      X1[i]=th;
      X2[i]=tm;
      Y[i]=(double)poidev(le*th*0.5);
    } // simulation

    for(i=0;i<TRIAL-1;i++) {
      sig2Xs[i]=(X1[i+1]-X1[i])*(X1[i+1]-X1[i])+(X2[i+1]-X2[i])*(X2[i+1]-X2[i]);
    }
    qsort(sig2Xs,TRIAL-1,sizeof(double),(int(*)(const void*,const void*))cmp);
    sig2X=sig2Xs[(TRIAL-1)/2];
    for(i=0;i<TRIAL-1;i++) {
      sig2Ys[i]=(Y[i+1]-Y[i])*(Y[i+1]-Y[i]);
    }
    qsort(sig2Ys,TRIAL-1,sizeof(double),(int(*)(const void*,const void*))cmp);
    sig2Y=sig2Ys[(TRIAL-1)/2];
    // fixing sigma sq. by median
    
    for(i=0;i<TRIAL;i++) ky(i)=kerY(Y[i],seg,sig2Y);

    
    rankXv=0;
    LRAKX(KX,pp,X1,X2,sig2X,rankX);
    
    Matrix RX(TRIAL,rankXv);
    for(i=0;i<TRIAL;i++) {
      for(j=0;j<rankXv;j++) {
	RX(pp[i],j)=KX[i+TRIAL*j];
      }
    }
 
    for(i=0;i<rankXv;i++) {
      mupia0[i]=0.0;
      for(j=0;j<TRIAL;j++) {
        mupia0[i]+=RX(j,i);
      }
    }
  
    for(i=0;i<TRIAL;i++) {
      mupia(i)=0.0;
      for(j=0;j<rankXv;j++) {
	mupia(i)+=RX(i,j)*mupia0[j];
      }
    }

    /*    
    rankXv=0;
    LRAKX(KX,pp,X1,X2,sig2X,rankX);
    
    Matrix RX(TRIAL,rankXv);
    for(i=0;i<TRIAL;i++) {
      for(j=0;j<rankXv;j++) {
	RX(pp[i],j)=KX[i+TRIAL*j];
      }
    }
  
    for(i=0;i<TRIAL;i++) {
      mupia(i)=0.0;
      for(j=0;j<TRIAL;j++) {
	mupia(i)+=KX[i+TRIAL*j];
      }
    }
    */

    Matrix idr2(rankXv,rankXv);
    for(i=0;i<rankXv;i++) {
      for(j=0;j<rankXv;j++) {
        if(i==j) idr2(i,i)=1;
        else idr2(i,j)=0;
      }
    }
    mua0=mupia/(TRIAL*EPS);
    mua=mua0-RX*((RX.transpose()*RX+TRIAL*EPS*idr2).inverse()*(RX.transpose()*mua0));
     //low rank approx.
    
    Matrix id(TRIAL,TRIAL);
    for(i=0;i<TRIAL;i++) {
      for(j=0;j<TRIAL;j++) {
        if(i==j) id(i,i)=1;
        else id(i,j)=0;
      }
    }
    for(i=0;i<TRIAL;i++) {
      for(j=0;j<TRIAL;j++) {
	//gx(i,j)=kerX(X1[i],X2[i],X1[j],X2[j],sig2X);
	gy(i,j)=kerY(Y[i],Y[j],sig2Y); 
      }
    }
    
    /*
    for(i=0;i<TRIAL;i++) {
      mupi(i)=0;
      for(j=0;j<TRIAL;j++) {
	mupi(i)+=gx(i,j);
      }
    } 
    //cout << "mupi" << endl << mupi;
    //cout << "mupia" << endl << mupia;
    ColumnVector mu=(gx+TRIAL*EPS*id).inverse()*mupi;
    // exact
    */

    //cout << "exact Gx" << endl; cout << gx;
    //cout << "approx. Gx" << endl; cout << RX*RX.transpose();
    // comparison with exact comp.  

    
    rankYv=0;   
    LRAKY(KY,pp,Y,sig2Y,rankY);

    Matrix RY(TRIAL,rankYv);
    for(i=0;i<TRIAL;i++) {
      for(j=0;j<rankYv;j++) {
	RY(pp[i],j)=KY[i+TRIAL*j];
      }
    }

    for(i=0;i<TRIAL;i++) {
      for(j=0;j<TRIAL;j++) {
        if(i==j) lam(i,i)=mua(i); 
        else lam(i,j)=0;
      }
    }
    Matrix idr(rankYv,rankYv);
    for(i=0;i<rankYv;i++) {
      for(j=0;j<rankYv;j++) {
        if(i==j) idr(i,i)=1;
        else idr(i,j)=0;
      }
    }
    
    Matrix lr=lam*RY;
    Matrix rl=RY.transpose()*lr;
 
    wea=lr*(rl*rl+DEL*idr).inverse()*lr.transpose()*ky;
     //low rank approx.
    
    /*    
    for(i=0;i<TRIAL;i++) lam(i,i)=mua(i); 
    Matrix ly=lam*gy;
    ColumnVector we=ly*(ly*ly+DEL*id).inverse()*lam*ky;

    //cout << "exact Gy" << endl; cout << gy;
    //cout << "approx. Gy" << endl; cout << RY*RY.transpose();
    //comparison with exact comp.
    */
 
    meanX1=varX1=0.0;
    for(i=0;i<TRIAL;i++) {
      meanX1+=wea(i)*X1[i];
      varX1+=wea(i)*X1[i]*X1[i];
      //meanX1+=we(i)*X1[i];
      //varX1+=we(i)*X1[i]*X1[i];
    } // we exact, wea low rank approx.  
    varX1-=meanX1*meanX1;

    meanX2=varX2=0.0;
    for(i=0;i<TRIAL;i++) {
      meanX2+=wea(i)*X2[i];
      varX2+=wea(i)*X2[i]*X2[i];
      //meanX2+=we(i)*X2[i];
      //varX2+=we(i)*X2[i]*X2[i];
    } // we exact, wea low rank approx.
    varX2-=meanX2*meanX2;

    printf("THETA mean %.3f var %.3f\n",meanX1,varX1);
    printf("TMRCA mean %.3f var %.3f\n",meanX2,varX2);
  
    avmeX1+=meanX1/LOOP;
    avmeX2+=meanX2/LOOP;
    mse+=(meanX1-THETA0)*(meanX1-THETA0)/LOOP+(meanX2-TMRCA0)*(meanX2-TMRCA0)/LOOP;
  }    
  
  printf("THETA %.3f TMRCA %.3f MSE %.6f\n",avmeX1,avmeX2,mse);

  free(KX);
  free(KY);

  free(sig2Xs);
  free(sig2Ys);
  //for fixing sigma sq. by median

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

double kerX(double xa1,double xb1,double xa2,double xb2,double sig2) {
  return exp(-0.5*((xa1-xa2)*(xa1-xa2)+(xb1-xb2)*(xb1-xb2))/sig2);
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

void LRAKX(long double G[],int pp[],double x1[],double x2[],double sig,int *lr) {

     int jast,i,j,iter;
     int n=TRIAL;
     long double a,b,residual,maxdiagG,residual0;
     long double *diagG;

     diagG=(long double*)calloc(n,sizeof(long double));
     
     iter=0;
     residual=TRIAL;
     residual0=TRIAL;

     for(i=0;i<=n-1;i++) pp[i]=i;
     for(i=0;i<=n-1;i++) diagG[i]=1;
     
     jast=0;

     while(residual>residual0*TOL&&iter<NMAX) {
       //       while(iter<NMAX) {

	  if(jast!=iter) {
	       i=pp[jast]; pp[jast]=pp[iter]; pp[iter]=i;
	       for(i=0;i<=iter;i++) {
		  a=G[jast+n*i];G[jast+n*i]=G[iter+n*i];
                  G[iter+n*i]=a;
     	       }
	  }
	  G[iter*(n+1)]=sqrt(diagG[jast]);
	  for(i=iter+1;i<=n-1;i++) {
	    G[i+n*iter]=exp(-0.5/sig*(pow(x1[pp[iter]]-x1[pp[i]],2.0)+pow(x2[pp[iter]]-x2[pp[i]],2.0)));
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
	  if(iter==1) residual0=residual;
	  //printf("KX %d s2 %.20Lf\n",iter,residual);

     }
     printf("KX %d s2 %.20Lf\n",iter,residual);

     *lr=iter;
     free(diagG);
     return;
}  

void LRAKY(long double G[],int pp[],double y[],double sig,int *lr) {

     int jast,i,j,iter;
     int n=TRIAL;
     long double a,b,residual,maxdiagG,residual0;
     long double *diagG;

     diagG=(long double*)calloc(n,sizeof(long double));
     
     iter=0;
     residual=TRIAL;
     residual0=TRIAL;

     for(i=0;i<=n-1;i++) pp[i]=i;
     for(i=0;i<=n-1;i++) diagG[i]=1;
	  
     jast=0;

         while(residual>TOL*residual0&&iter<NMAX) {
	   //	        while(iter<NMAX) {

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
	  if(iter==1) residual0=residual;
	  //printf("KY %d s1 %.20Lf\n",iter,residual);
	  	  
     }
		printf("KY %d s1 %.20Lf\n",iter,residual);
     *lr=iter;     
     free(diagG);
     return;

}  
