#ifndef KNNRING_H
#define KNNRING_H


// Definition of the kNN result 
typedef struct knnresult
{
	int* nidx; //!< Indices (0-based) of nearest neighbors [m-by-k]
	double* ndist;	//!< Distance of nearest neighbors[m-by-k]
	int m;	//!< Number of query points[scalar]
	int k;	//!< Number of nearest neighbors[scalar]
} knnresult;

knnresult distrAllkNN(double* X ,int n,int d,int k);

knnresult init_knnresult(int m_arg , int k_arg );

knnresult kNN(double* X,double* Y,int n,int m,int d,int k);

double * edm(double *a , double *b , int n , int m ,int d   );

double * sub_array(int n , int d , double *temp);

int pick_index(double *X , double query , int n );


#endif