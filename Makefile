# Makefile

EXE=d2q9-bgk

# CC=gcc
CC=icc
CFLAGS= -std=c99 -Wall -Ofast -mtune=native -march=native -fma -xHOST -align -qopt-assume-safe-padding -qopenmp
LIBS = -lm

MPI_CC=mpiicc 
MPI_CFLAGS=-std=c99 -Wall
MPI_EXE=mpi_test
MPI_NUM_PROCESSES=4

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
# REF_FINAL_STATE_FILE=check/256x256.final_state.dat
# REF_AV_VELS_FILE=check/256x256.av_vels.dat
REF_FINAL_STATE_FILE=check/1024x1024.final_state.dat
REF_AV_VELS_FILE=check/1024x1024.av_vels.dat

# export OMP_PROC_BIND=true
# export OMP_PLACES=cores
# export OMP_NUM_THREADS=28

all: $(EXE)

$(EXE): $(EXE).c
	$(MPI_CC) $(CFLAGS) $^ $(LIBS) -o $@

mpi_test: mpi_test.c
	$(MPI_CC) $(MPI_CFLAGS) $^ -o $(MPI_EXE)

mpirun: mpi_test
	mpiexec -np $(MPI_NUM_PROCESSES) ./$(MPI_EXE)

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean mpi_test mpirun

clean:
	rm -f $(EXE) $(MPI_EXE)
