# Makefile

EXE=d2q9-bgk

# CC=gcc
CC=icc
CFLAGS= -std=c99 -Wall -Ofast -mtune=native -march=native -fma -xHOST -qopenmp -qopt-report=5 -qopt-report-phase=openmp -qopt-report-file=report.txt -align -qopt-assume-safe-padding
LIBS = -lm

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/128x128.final_state.dat
REF_AV_VELS_FILE=check/128x128.av_vels.dat

# export OMP_PROC_BIND=true
# export OMP_PLACES=cores
# export OMP_NUM_THREADS=28

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE)
