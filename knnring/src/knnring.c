#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef NAN
/* NAN is supported */
#endif
#include <unistd.h>
#include <cblas.h>
#include "quickselect.h"
#include "knnring.h"

knnresult init_knnresult(int m_arg , int k_arg )
{
	knnresult result;
	result.nidx = (int *) malloc(sizeof(int)*m_arg*k_arg);
	result.ndist = (double *) malloc(sizeof(double)*m_arg*k_arg);
	result.m =m_arg;
	result.k=k_arg;
	return result;
}


knnresult kNN(double* X,double* Y,int n,int m,int d,int k)
{
	knnresult result;
	result = init_knnresult(m,k);
	double *distances =  (double *) malloc(n*m*sizeof(double));
	double *orig_dist =  (double *) malloc(n*sizeof(double));
	
	distances = edm(X,Y,n,m,d);

	for (int i = 0; i < m; i++)
	{
		cblas_dcopy(n,&distances[i*n] , 1 , orig_dist,1);
		for (int j = 0; j < k; j++)
		{

			result.ndist[i*k+j] = quickselect( distances ,  i*n+j , (i+1)*n-1 , i*n+j );
			result.nidx[i*k+j] = pick_index(orig_dist,result.ndist[i*k+j],n);
		}
	}
	free(distances);
	free(orig_dist);
	return result;
}


double * edm(double *a , double *b , int n , int m ,int d   )
{

	double alpha,beta;
	alpha=1.0;
	beta=0.0;

	double *c ,*temp1, *temp2;
	double  *temp3;
	c = (double *) malloc(n*m*sizeof(double));
	temp3 = (double *) malloc(m*n*sizeof(double));
	temp1 = (double *) malloc(n*sizeof(double));
	temp2 =  (double *) malloc(m*sizeof(double));

	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,m,d,alpha,a,d,b,d,beta,c,m);	
	temp1 = sub_array(n,d,a);
	temp2 = sub_array(m,d,b);
	
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			if(i==j){
				if( temp1[i]-(2*c[i*m+j])+temp2[j]<=0.0001 || sqrt(temp1[i]-(2*c[i*m+j])+temp2[j])==sqrt(-0) ){
	 					
					temp3[j*n+i]==0.0;
				}
				else	
				{
					temp3[j*n+i] = sqrt(temp1[i]-(2*c[i*m+j])+temp2[j]);
				}
			}
			else{
				temp3[j*n+i] = sqrt(temp1[i]-(2*c[i*m+j])+temp2[j]);
			}		
		}
	}
	free(c);
	free(temp1);
	free(temp2);
	return temp3;
}

double * sub_array(int n , int d , double *temp)
{
	double *c;
	c = (double *) malloc(n*sizeof(double));
	for (int i = 0; i < n ; i++)
	{
		c[i] = pow(cblas_dnrm2( d , &temp[i*d], 1),2);
	}
	return c;
}

int pick_index(double *X , double query , int n )
{
	for (int i = 0; i < n; i++)
	{
		if(X[i] == query )
			return i;
	}
}

