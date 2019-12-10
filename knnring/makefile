# define the shell to bash
SHELL := /bin/bash

IDIR =./inc

ODIR=./lib

SDIR=./src

_OBJ = knnring.o knnring_blocking.o knnring_no_blocking.o 
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

_SRC = knnring.c knnring_blocking.c knnring_no_blocking.c 
SRC = $(patsubst %,$(SDIR)/%,$(_SRC))

_DEPS = quickselect.h knnring.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_FINAL = knnring_sequential.a knnring_mpi.a 
FINAL = $(patsubst %,$(ODIR)/%,$(_FINAL))

# define the C/C++ compiler to use,default here is clang
CC = gcc
MPICC = mpicc
CFLAGS = -lm -L$(NETLIB_LAPACK_ROOT)/lib64 -lcblas -I$(NETLIB_LAPACK_ROOT)/include -I$(IDIR)

all: 
	$(CC) -c src/knnring.c -o lib/knnring.o $(CFLAGS)
	$(MPICC) -c src/knnring_no_blocking.c -o lib/knnring_mpi.o $(CFLAGS)
	ar rcs lib/knnring_sequential.a lib/knnring.o
	ar rcs lib/knnring_mpi.a  lib/knnring_mpi.o
