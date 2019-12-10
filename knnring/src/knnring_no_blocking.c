#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cblas.h>
#include "quickselect.h"
#include "knnring.h"

int update(double *new, double *old, int *new_i , int *old_i ,int m ,int k);
int sort_d(double *array, double addition , int possition , int n , int offset);
int sort_i(int *array, int addition , int possition , int n , int offset);
double *d_copy(double *a , double *b, int n,int d);
int correct_ind(int *index, int n, int k, int offset);
int shift(int *a , int n);


knnresult distrAllkNN(double* X ,int n,int d,int k)
{
	// Find out rank, size
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int next,prev;

	MPI_Status status[2];
	MPI_Request reqs[2];

	knnresult f_data , helper , real;

	f_data = init_knnresult( n ,  k );
	helper = init_knnresult( n ,  k );

	double *static_data , *dynamic_data , *temp_data;
	static_data = d_copy(X,static_data,n,d);

	int support[world_size];
	
	if(world_rank==0)
	{
		next=1;
		prev=world_size-1;
	}
	else if(world_rank==world_size-1)
	{
		next=0;
		prev=world_size-2;
	}
	else
	{
		next=world_rank+1;
		prev=world_rank-1;
	}

	for (int i = 0; i < world_size; i++)
	{
		if(i==0)
		{
			support[i]=world_size-1;
		}
		else
		{
			support[i]=i-1;
		}
	}

	for(int i=0;i<world_size;i++)
	{
    	if(i==0)
    	{
    		dynamic_data = d_copy(static_data,dynamic_data,n,d);
    		temp_data = d_copy(dynamic_data,temp_data,n,d);

            MPI_Isend(static_data ,n*d , MPI_DOUBLE , next , 0 , MPI_COMM_WORLD   ,&reqs[0]);
            MPI_Irecv(dynamic_data , n*d, MPI_DOUBLE,prev,0,MPI_COMM_WORLD , &reqs[1] );
            
            f_data = kNN(temp_data,static_data  , n , n , d ,k);
    		correct_ind(f_data.nidx, n, k, support[world_rank]*n);
    	}
        else if (i==world_size-1)
        {
        	MPI_Waitall(2 , reqs , status);

    		helper = kNN(dynamic_data, static_data , n , n , d , k);
    		shift(support,world_size);
    		correct_ind(helper.nidx, n, k, support[world_rank]*n);
    		update(helper.ndist , f_data.ndist , helper.nidx , f_data.nidx, n , k );
        }
		else{

			MPI_Waitall(2 , reqs , status);


        	temp_data = d_copy(dynamic_data,temp_data,n,d);
            
            MPI_Isend(temp_data ,n*d , MPI_DOUBLE , next , 0 , MPI_COMM_WORLD,&reqs[0]);
            MPI_Irecv(dynamic_data , n*d, MPI_DOUBLE,prev,0,MPI_COMM_WORLD,&reqs[1]);

            helper = kNN(temp_data, static_data , n , n , d , k);
    		shift(support,world_size);
    		correct_ind(helper.nidx, n, k, support[world_rank]*n); 
            update(helper.ndist , f_data.ndist , helper.nidx , f_data.nidx, n , k ); 
        }      
    }
    free(dynamic_data);
    free(static_data);
    free(temp_data);
	return f_data;
}

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
	
	int counter=0;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			if(i==j){
				if( temp1[i]-(2*c[i*m+j])+temp2[j]<=0.0001 || sqrt(temp1[i]-(2*c[i*m+j])+temp2[j])==sqrt(-0) ){
	 					
					temp3[j*n+i]==0.0;
					counter++;				
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

int update(double *new, double *old, int *new_i , int *old_i ,int m ,int k)
{
	for (int i = 0; i < m; i++)
	{
		int counter=0;
        for (int j = 0; j < k; j++)
        {
            if(new[i*k+counter] < old[i*k+j] && new_i[i*k+counter] != old_i[i*k+j] )
			{
				sort_d(old , new[i*k+counter] , i*k+j , k , i*k );
            	sort_i(old_i,new_i[i*k+counter] , i*k+j , k , i*k );
            	counter++;
            }
        }
	}
	return 1;
}

double *d_copy(double *a , double *b, int n,int d)
{
	b = (double *) malloc(sizeof(double)*n*d);
	for(int i=0; i<n*d ; i++)
	{
		b[i] = a[i];
	}
	return b;
}

int sort_d(double *array, double addition , int possition , int n , int offset)
{
	double temp1,temp2;
	for (int i = 0; i < n; i++)
	{
		if(offset+i==possition)
		{
			temp1 = array[offset+i];
			array[offset+i] = addition;
		}
		else if(offset+i>possition)
		{
			temp2 = array[offset+i];
			array[offset+i] = temp1;
			temp1 = temp2;
		}
	}
	return 1;
}

int sort_i(int *array, int addition , int possition , int n , int offset)
{
	int temp1,temp2;
	for (int i = 0; i < n; i++)
	{
		if(offset+i==possition)
		{
			temp1 = array[offset+i];
			array[offset+i] = addition;
		}
		else if(offset+i>possition)
		{
			temp2 = array[offset+i];
			array[offset+i] = temp1;
			temp1 = temp2;
		}
	}
	return 1;
}

int correct_ind(int *index, int n, int k, int offset)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < k; j++)
		{
			index[i*k+j] = index[i*k+j] + offset;
		}
	}
	return 1;
}

int shift(int* a , int n)
{
	int temp1 , temp2;
	for (int i = 0; i < n; i++)
	{
		if(i==0)
		{
			temp1 = a[i];
			a[i] = a[n-1];
		}
		else
		{
			temp2 = a[i];
			a[i] = temp1;
			temp1 = temp2;
		}
	}
}