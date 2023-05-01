# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.
module load languages/anaconda2/5.0.1
# module load icc/2017.1.132-GCC-5.4.0-2.26
module load languages/intel/2020-u4
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS=28