/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <string.h>
#include <malloc.h>
#include <mpi.h>
#include <assert.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;


/* struct to hold the 'speed' values */
typedef struct
{
  float *restrict v0;
  float *restrict v1;
  float *restrict v2;
  float *restrict v3;
  float *restrict v4;
  float *restrict v5;
  float *restrict v6;
  float *restrict v7;
  float *restrict v8;
} t_speed_soa;



/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/* Allocate memory for a t_speed_soa structure */
t_speed_soa* alloc_t_speed_soa(const int size)
{
  t_speed_soa* speeds = (t_speed_soa*) _mm_malloc(sizeof(t_speed_soa), 64);
  speeds->v0 = (float*) _mm_malloc(sizeof(float) * size + 64, 64);
  speeds->v1 = (float*) _mm_malloc(sizeof(float) * size + 64, 64);
  speeds->v2 = (float*) _mm_malloc(sizeof(float) * size + 64, 64);
  speeds->v3 = (float*) _mm_malloc(sizeof(float) * size + 64, 64);
  speeds->v4 = (float*) _mm_malloc(sizeof(float) * size + 64, 64);
  speeds->v5 = (float*) _mm_malloc(sizeof(float) * size + 64, 64);
  speeds->v6 = (float*) _mm_malloc(sizeof(float) * size + 64, 64);
  speeds->v7 = (float*) _mm_malloc(sizeof(float) * size + 64, 64);
  speeds->v8 = (float*) _mm_malloc(sizeof(float) * size + 64, 64);

  // if any alloc failed, die
  if (speeds->v0 == NULL || speeds->v1 == NULL || speeds->v2 == NULL || speeds->v3 == NULL || speeds->v4 == NULL || speeds->v5 == NULL || speeds->v6 == NULL || speeds->v7 == NULL || speeds->v8 == NULL)
  {
    die("Memory allocation failed", __LINE__, __FILE__);
  }

  return speeds;
}

/* Free memory for a t_speed_soa structure */
void free_t_speed_soa(t_speed_soa* speeds)
{
  _mm_free(speeds->v0);
  _mm_free(speeds->v1);
  _mm_free(speeds->v2);
  _mm_free(speeds->v3);
  _mm_free(speeds->v4);
  _mm_free(speeds->v5);
  _mm_free(speeds->v6);
  _mm_free(speeds->v7);
  _mm_free(speeds->v8);
  _mm_free(speeds);
}

#define _ASSUME_ALIGNED_SOA(soa) \
  __assume_aligned(soa->v0, 64); \
  __assume_aligned(soa->v1, 64); \
  __assume_aligned(soa->v2, 64); \
  __assume_aligned(soa->v3, 64); \
  __assume_aligned(soa->v4, 64); \
  __assume_aligned(soa->v5, 64); \
  __assume_aligned(soa->v6, 64); \
  __assume_aligned(soa->v7, 64); \
  __assume_aligned(soa->v8, 64);

#define _ASSUME_PARAMS(params) \
  __assume((params.nx % 2) == 0); \
  __assume((params.nx % 4) == 0); \
  __assume((params.nx % 8) == 0); \
  __assume((params.nx % 16) == 0); \
  __assume((params.nx % 32) == 0); \
  __assume((params.nx % 64) == 0); \
  __assume((params.nx % 128) == 0); \
  // __assume((params.ny % 2) == 0); \
  // __assume((params.ny % 4) == 0); \
  // __assume((params.ny % 8) == 0); \
  // __assume((params.ny % 16) == 0); \
  // __assume((params.ny % 32) == 0); \
  // __assume((params.ny % 64) == 0); \
  // __assume((params.ny % 128) == 0); \


#define SOA_SUM(soa, idx) (soa->v0[idx] + soa->v1[idx] + soa->v2[idx] + soa->v3[idx] + soa->v4[idx] + soa->v5[idx] + soa->v6[idx] + soa->v7[idx] + soa->v8[idx])
#define SOA_SET_ALL(soa, idx, val) \
  soa->v0[idx] = val; \
  soa->v1[idx] = val; \
  soa->v2[idx] = val; \
  soa->v3[idx] = val; \
  soa->v4[idx] = val; \
  soa->v5[idx] = val; \
  soa->v6[idx] = val; \
  soa->v7[idx] = val; \
  soa->v8[idx] = val;



/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed_soa** start_cells_ptr, t_speed_soa** tmp_cells_ptr, t_speed_soa** end_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int rank, int size);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int accelerate_flow(const t_param params, t_speed_soa*restrict cells, int*restrict obstacles, int row_to_accelerate);
int propagate(const t_param params, t_speed_soa*restrict cells, t_speed_soa*restrict tmp_cells);
float collision(const t_param params, t_speed_soa*restrict start_cells, t_speed_soa*restrict cells, t_speed_soa*restrict tmp_cells, int*restrict obstacles, int rowcount);
int write_values(const t_param params, t_speed_soa* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed_soa** start_cells_ptr, t_speed_soa** tmp_cells_ptr, t_speed_soa** end_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed_soa* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed_soa* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed_soa* cells, int* obstacles);


inline void swap_boundaries(t_param params, t_speed_soa* cells, int rowcount, int left, int right, float* sendbuf, float* recvbuf) {
  int tag = 0;           /* scope for adding extra information to a message */
  MPI_Status status;     /* struct used by MPI_Recv */

  const int row_buf_size = params.nx * sizeof(float);

  // Copy row[1] to sendbuf
  memcpy(sendbuf + 0*params.nx, cells->v0 + params.nx, row_buf_size);
  memcpy(sendbuf + 1*params.nx, cells->v1 + params.nx, row_buf_size);
  memcpy(sendbuf + 2*params.nx, cells->v2 + params.nx, row_buf_size);
  memcpy(sendbuf + 3*params.nx, cells->v3 + params.nx, row_buf_size);
  memcpy(sendbuf + 4*params.nx, cells->v4 + params.nx, row_buf_size);
  memcpy(sendbuf + 5*params.nx, cells->v5 + params.nx, row_buf_size);
  memcpy(sendbuf + 6*params.nx, cells->v6 + params.nx, row_buf_size);
  memcpy(sendbuf + 7*params.nx, cells->v7 + params.nx, row_buf_size);
  memcpy(sendbuf + 8*params.nx, cells->v8 + params.nx, row_buf_size);

  // Send top row to the left and receive from the right
  MPI_Sendrecv(sendbuf, params.nx * 9, MPI_FLOAT, left, tag,
               recvbuf, params.nx * 9, MPI_FLOAT, right, tag,
               MPI_COMM_WORLD, &status);

  // Copy received data to row[n-1]
  memcpy(cells->v0 + (rowcount + 1) * params.nx, recvbuf + 0*params.nx, row_buf_size);
  memcpy(cells->v1 + (rowcount + 1) * params.nx, recvbuf + 1*params.nx, row_buf_size);
  memcpy(cells->v2 + (rowcount + 1) * params.nx, recvbuf + 2*params.nx, row_buf_size);
  memcpy(cells->v3 + (rowcount + 1) * params.nx, recvbuf + 3*params.nx, row_buf_size);
  memcpy(cells->v4 + (rowcount + 1) * params.nx, recvbuf + 4*params.nx, row_buf_size);
  memcpy(cells->v5 + (rowcount + 1) * params.nx, recvbuf + 5*params.nx, row_buf_size);
  memcpy(cells->v6 + (rowcount + 1) * params.nx, recvbuf + 6*params.nx, row_buf_size);
  memcpy(cells->v7 + (rowcount + 1) * params.nx, recvbuf + 7*params.nx, row_buf_size);
  memcpy(cells->v8 + (rowcount + 1) * params.nx, recvbuf + 8*params.nx, row_buf_size);

  //////////////////////////////////////////////////////////////////////////////////////

  // Copy row[n-2] to sendbuf
  memcpy(sendbuf + 0*params.nx, cells->v0 + rowcount * params.nx, row_buf_size);
  memcpy(sendbuf + 1*params.nx, cells->v1 + rowcount * params.nx, row_buf_size);
  memcpy(sendbuf + 2*params.nx, cells->v2 + rowcount * params.nx, row_buf_size);
  memcpy(sendbuf + 3*params.nx, cells->v3 + rowcount * params.nx, row_buf_size);
  memcpy(sendbuf + 4*params.nx, cells->v4 + rowcount * params.nx, row_buf_size);
  memcpy(sendbuf + 5*params.nx, cells->v5 + rowcount * params.nx, row_buf_size);
  memcpy(sendbuf + 6*params.nx, cells->v6 + rowcount * params.nx, row_buf_size);
  memcpy(sendbuf + 7*params.nx, cells->v7 + rowcount * params.nx, row_buf_size);
  memcpy(sendbuf + 8*params.nx, cells->v8 + rowcount * params.nx, row_buf_size);
  

  MPI_Sendrecv(sendbuf, params.nx * 9, MPI_FLOAT, right, tag,
               recvbuf, params.nx * 9, MPI_FLOAT, left, tag,
               MPI_COMM_WORLD, &status);

  // Copy received data to row[0]
  memcpy(cells->v0, recvbuf + 0*params.nx, row_buf_size);
  memcpy(cells->v1, recvbuf + 1*params.nx, row_buf_size);
  memcpy(cells->v2, recvbuf + 2*params.nx, row_buf_size);
  memcpy(cells->v3, recvbuf + 3*params.nx, row_buf_size);
  memcpy(cells->v4, recvbuf + 4*params.nx, row_buf_size);
  memcpy(cells->v5, recvbuf + 5*params.nx, row_buf_size);
  memcpy(cells->v6, recvbuf + 6*params.nx, row_buf_size);
  memcpy(cells->v7, recvbuf + 7*params.nx, row_buf_size);
  memcpy(cells->v8, recvbuf + 8*params.nx, row_buf_size);
}



/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed_soa *start_cells = NULL;  /* grid containing fluid densities */
  t_speed_soa *tmp_cells = NULL;    /* scratch space */
  t_speed_soa *end_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */


  // MPI Setup
  MPI_Status status;
  int mpi_rank, mpi_size, tag = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;

  // This is executed for all ranks and will load only the data for the rows that are relevant for this rank
  // Note: we will have two extra rows, one at the top and one at the bottom, that will be used for communication (the halo)

  initialise(paramfile, obstaclefile, &params, &start_cells, &tmp_cells, &end_cells, &obstacles, &av_vels, mpi_rank, mpi_size);

  int left_rank = (mpi_rank == 0)? mpi_size - 1 : mpi_rank - 1;
  int right_rank = (mpi_rank + 1) % mpi_size;

  // Calculate inclusive start row and exclusive end row 
  int base_rowcount = params.ny / mpi_size;
  int leftover_rows = params.ny % mpi_size;
  int rowcount = (mpi_rank < leftover_rows) ? base_rowcount + 1 : base_rowcount;

  int start_row;
  if (mpi_rank < leftover_rows) {
      start_row = mpi_rank * (base_rowcount + 1);
  } else {
      start_row = (leftover_rows * (base_rowcount + 1)) + (mpi_rank - leftover_rows) * base_rowcount;
  }
  int end_row = start_row + rowcount;
  printf("Rowcount: %d, neighbours: l:%d,r:%d, start: %d, end: %d\n", rowcount, left_rank, right_rank, start_row, end_row);

  // colcount = params.nx; (same as before MPI)

  // Allocate buffers for sending and receiving
  // buffer size: one row of cells, each cell contains 9 floats
  // = colcount * 9 * float
  float *sendbuf = (float*)malloc(params.nx * 9 * sizeof(float));
  float *recvbuf = (float*)malloc(params.nx * 9 * sizeof(float));

  // float num_cells_without_obstacles = 0;
  // _ASSUME_PARAMS(params);
  // __assume_aligned(obstacles, 64);
  // // #pragma omp parallel for reduction(+:num_cells_without_obstacles)
  // for (int jj = 0; jj < rowcount; jj++)
  // {
  //   const int jj_ = jj * params.nx;
  //   #pragma omp simd reduction(+:num_cells_without_obstacles)
  //   for (int ii = 0; ii < params.nx; ii++)
  //   {
  //     const int cell_idx = ii + jj_;
  //     if (!obstacles[cell_idx])
  //     {
  //       num_cells_without_obstacles++;
  //     }
  //   }
  // }

  // printf("Num obstacles: %d\n", params.nx * rowcount - (int)num_cells_without_obstacles);


  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  // only allow the last rank to accelerate
  // This only works if we have fewer than params.ny / 2 ranks
  assert(mpi_size <= params.ny / 2);
  int is_accelerator = (mpi_rank == mpi_size - 1);

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    float tot_vel = 0.0f;
    if (is_accelerator) accelerate_flow(params, start_cells, obstacles, rowcount - 1);
    // Distribute data to neighbours
    swap_boundaries(params, start_cells, rowcount, left_rank, right_rank, sendbuf, recvbuf);
    /* loop over _all_ cells */
    // propagate(params, start_cells, tmp_cells);
    tot_vel = collision(params, start_cells, end_cells, tmp_cells, obstacles, rowcount);
    av_vels[tt] = tot_vel;

    // swap pointers
    t_speed_soa* tmp = start_cells;
    start_cells = end_cells;
    end_cells = tmp;
  }
  // printf("Finished computing\n");
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 
  /*
  int MPI_Gatherv(
    const void *sendbuf, 
    int sendcount, 
    MPI_Datatype sendtype, 
    void *recvbuf, 
    const int *recvcounts, 
    const int *displs, 
    MPI_Datatype recvtype, 
    int root, 
    MPI_Comm comm
  ); 
  */
  // Gather all speeds (array big enough to hold all cells across all ranks)
  t_speed_soa *final_cells = alloc_t_speed_soa(params.nx * params.ny);
  int *recvcounts = NULL;
  int *displs = NULL;
  if (mpi_rank == 0) {
    // calculate recvcounts and displs
    recvcounts = (int*)malloc(mpi_size * sizeof(int));
    displs = (int*)malloc(mpi_size * sizeof(int));
    for (int i = 0; i < mpi_size; i++) {
      recvcounts[i] = params.nx * ((i < leftover_rows) ? base_rowcount + 1 : base_rowcount);
      displs[i] = params.nx * ((i < leftover_rows) ? (base_rowcount + 1) * i : (leftover_rows * (base_rowcount + 1)) + (i - leftover_rows) * base_rowcount);
    }
  }

  // printf("gathering\n");
  MPI_Gatherv(start_cells->v0 + params.nx, params.nx * rowcount, MPI_FLOAT, final_cells->v0, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(start_cells->v1 + params.nx, params.nx * rowcount, MPI_FLOAT, final_cells->v1, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(start_cells->v2 + params.nx, params.nx * rowcount, MPI_FLOAT, final_cells->v2, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(start_cells->v3 + params.nx, params.nx * rowcount, MPI_FLOAT, final_cells->v3, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(start_cells->v4 + params.nx, params.nx * rowcount, MPI_FLOAT, final_cells->v4, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(start_cells->v5 + params.nx, params.nx * rowcount, MPI_FLOAT, final_cells->v5, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(start_cells->v6 + params.nx, params.nx * rowcount, MPI_FLOAT, final_cells->v6, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(start_cells->v7 + params.nx, params.nx * rowcount, MPI_FLOAT, final_cells->v7, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(start_cells->v8 + params.nx, params.nx * rowcount, MPI_FLOAT, final_cells->v8, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // printf("gathered\n");
  // Gather all obstacles (array big enough to hold all cells across all ranks)
  // TODO: do we need to do this? Or just pre-load the obstacles on rank 0?
  int *final_obstacles = _mm_malloc(params.nx * params.ny * sizeof(int), 64);
  // MPI_Gather(obstacles, size, MPI_INT, final_obstacles, size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(obstacles, rowcount * params.nx, MPI_INT, final_obstacles, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
  // printf("gathered obstacles\n");

  // Gather all av_vels
  float *final_av_vels = malloc(params.maxIters * sizeof(float) * mpi_size);
  MPI_Gather(av_vels, params.maxIters, MPI_FLOAT, final_av_vels, params.maxIters, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // printf("gathered av_vels\n");




  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
  if (mpi_rank == 0) {
    int num_final_obstacles = 0;
    for (int ii = 0; ii < params.nx * params.ny; ii++) {
      if (final_obstacles[ii] == 1) num_final_obstacles++;
    }
    const int final_num_cells_without_obstacles = (params.nx * params.ny) - num_final_obstacles;

    // calculate final av_vels
    float final_av_vel = 0.0f;
    for (int ii = 0; ii < params.maxIters; ii++) {
      float total_step_vel = 0.0f;
      for (int jj = 0; jj < mpi_size; jj++) {
        total_step_vel += final_av_vels[ii + jj * params.maxIters];
      }
      av_vels[ii] = total_step_vel / final_num_cells_without_obstacles;
    }
    printf("Final number of obstacles: %d\n", num_final_obstacles);



    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, final_cells, final_obstacles));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
    write_values(params, final_cells, final_obstacles, av_vels);
    
    _mm_free(final_cells);
    _mm_free(final_obstacles);
  }

  // Clean up MPI
  MPI_Finalize();
  free(sendbuf);
  free(recvbuf);
  
  finalise(&params, &start_cells, &tmp_cells, &end_cells, &obstacles, &av_vels);
  
  return EXIT_SUCCESS;
}


int accelerate_flow(const t_param params, t_speed_soa*restrict cells, int*restrict obstacles, int row_to_accelerate)
{
  _ASSUME_ALIGNED_SOA(cells)
  __assume_aligned(obstacles, 64);
  
  _ASSUME_PARAMS(params)
  /* compute weighting factors */
  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  const int jj = row_to_accelerate;

  const t_speed_soa cells_ = *cells;

  #pragma omp simd
  for (int ii = 0; ii < params.nx; ii++)
  {
    const int idx = ii + jj * params.nx;
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + (jj - 1)*params.nx]
        && (cells_.v3[idx] - w1) > 0.f
        && (cells_.v6[idx] - w2) > 0.f
        && (cells_.v7[idx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells_.v1[idx] += w1;
      cells_.v5[idx] += w2;
      cells_.v8[idx] += w2;
      /* decrease 'west-side' densities */
      cells_.v3[idx] -= w1;
      cells_.v6[idx] -= w2;
      cells_.v7[idx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

int propagate(const t_param params, t_speed_soa*restrict cells, t_speed_soa*restrict tmp_cells)
{
  _ASSUME_ALIGNED_SOA(cells)
  _ASSUME_ALIGNED_SOA(tmp_cells)
  _ASSUME_PARAMS(params)
  
  // #pragma omp parallel for
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    { 
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = (jj + 1);
      if (y_n == params.ny) y_n = 0;
      int x_e = (ii + 1);
      if (x_e == params.nx) x_e = 0;
      const int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

      const int idx = ii + jj*params.nx;
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells->v0[idx] = cells->v0[idx]; /* central cell, no movement */
      tmp_cells->v1[idx] = cells->v1[x_w + jj*params.nx]; /* east */
      tmp_cells->v2[idx] = cells->v2[ii + y_s*params.nx]; /* north */
      tmp_cells->v3[idx] = cells->v3[x_e + jj*params.nx]; /* west */
      tmp_cells->v4[idx] = cells->v4[ii + y_n*params.nx]; /* south */
      tmp_cells->v5[idx] = cells->v5[x_w + y_s*params.nx]; /* north-east */
      tmp_cells->v6[idx] = cells->v6[x_e + y_s*params.nx]; /* north-west */
      tmp_cells->v7[idx] = cells->v7[x_e + y_n*params.nx]; /* south-west */
      tmp_cells->v8[idx] = cells->v8[x_w + y_n*params.nx]; /* south-east */
    }
  }
  return EXIT_SUCCESS;
}


float collision(const t_param params, t_speed_soa*restrict start_cells, t_speed_soa*restrict cells, t_speed_soa*restrict tmp_cells, int*restrict obstacles, int rowcount)
{
  _ASSUME_ALIGNED_SOA(cells)
  _ASSUME_ALIGNED_SOA(tmp_cells)
  __assume_aligned(obstacles, 64);
  _ASSUME_PARAMS(params)

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  float tot_u = 0.f; 

  const t_speed_soa start_cells_ = *start_cells;
  const t_speed_soa tmp_cells_ = *tmp_cells;
  const t_speed_soa cells_ = *cells;

  // #pragma omp parallel for reduction(+:tot_u)
  for (int jj = 1; jj < rowcount+1; jj++)
  {
    // y-boundaries wraparound is avoided by the halo cells
    const int jjnx = jj*params.nx;
    const int y_n = (jj + 1) * params.nx;
    const int y_s = (jj - 1) * params.nx;
    #pragma omp simd reduction(+:tot_u)
    for (int ii = 0; ii < params.nx; ii++)
    { 
      const int idx = ii + jjnx;
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      // const int y_n = (jj_ny) ? 0 : (jj + 1);
      const int x_e = (ii + 1) % params.nx;
      // const int y_s = jj_zero ? (jj + params.ny - 1) : (jj - 1);
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells_.v0[idx] = start_cells_.v0[idx]; /* central cell, no movement */
      tmp_cells_.v1[idx] = start_cells_.v1[x_w + jjnx]; /* east */
      tmp_cells_.v2[idx] = start_cells_.v2[ii + y_s]; /* north */
      tmp_cells_.v3[idx] = start_cells_.v3[x_e + jjnx]; /* west */
      tmp_cells_.v4[idx] = start_cells_.v4[ii + y_n]; /* south */
      tmp_cells_.v5[idx] = start_cells_.v5[x_w + y_s]; /* north-east */
      tmp_cells_.v6[idx] = start_cells_.v6[x_e + y_s]; /* north-west */
      tmp_cells_.v7[idx] = start_cells_.v7[x_e + y_n]; /* south-west */
      tmp_cells_.v8[idx] = start_cells_.v8[x_w + y_n]; /* south-east */

      /* don't consider occupied cells */
      if (obstacles[ii + (jj - 1)*params.nx])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        cells_.v1[idx] = tmp_cells_.v3[idx];
        cells_.v2[idx] = tmp_cells_.v4[idx];
        cells_.v3[idx] = tmp_cells_.v1[idx];
        cells_.v4[idx] = tmp_cells_.v2[idx];
        cells_.v5[idx] = tmp_cells_.v7[idx];
        cells_.v6[idx] = tmp_cells_.v8[idx];
        cells_.v7[idx] = tmp_cells_.v5[idx];
        cells_.v8[idx] = tmp_cells_.v6[idx];
      } else {
        /* compute local density total */
          /* compute local density total */
        const float local_density = tmp_cells_.v0[idx]
                            + tmp_cells_.v1[idx]
                            + tmp_cells_.v2[idx]
                            + tmp_cells_.v3[idx]
                            + tmp_cells_.v4[idx]
                            + tmp_cells_.v5[idx]
                            + tmp_cells_.v6[idx]
                            + tmp_cells_.v7[idx]
                            + tmp_cells_.v8[idx];

        /* compute x velocity component */
        const float u_x = (tmp_cells_.v1[idx]
                      + tmp_cells_.v5[idx]
                      + tmp_cells_.v8[idx]
                      - (tmp_cells_.v3[idx]
                          + tmp_cells_.v6[idx]
                          + tmp_cells_.v7[idx]))
                      / local_density;
        /* compute y velocity component */
        const float u_y = (tmp_cells_.v2[idx]
                      + tmp_cells_.v5[idx]
                      + tmp_cells_.v6[idx]
                      - (tmp_cells_.v4[idx]
                          + tmp_cells_.v7[idx]
                          + tmp_cells_.v8[idx]))
                      / local_density;
        /* velocity squared */
        const float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                    * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                          + (u[1] * u[1]) / (2.f / 9.f)
                                          - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                          + (u[2] * u[2]) / (2.f / 9.f)
                                          - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                          + (u[3] * u[3]) / (2.f / 9.f)
                                          - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                          + (u[4] * u[4]) / (2.f / 9.f)
                                          - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                          + (u[5] * u[5]) / (2.f / 9.f)
                                          - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                          + (u[6] * u[6]) / (2.f / 9.f)
                                          - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                          + (u[7] * u[7]) / (2.f / 9.f)
                                          - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                          + (u[8] * u[8]) / (2.f / 9.f)
                                          - u_sq / (2.f * c_sq));

        /* relaxation step */
        cells_.v0[idx] = tmp_cells_.v0[idx] + params.omega * (d_equ[0] - tmp_cells_.v0[idx]);
        cells_.v1[idx] = tmp_cells_.v1[idx] + params.omega * (d_equ[1] - tmp_cells_.v1[idx]);
        cells_.v2[idx] = tmp_cells_.v2[idx] + params.omega * (d_equ[2] - tmp_cells_.v2[idx]);
        cells_.v3[idx] = tmp_cells_.v3[idx] + params.omega * (d_equ[3] - tmp_cells_.v3[idx]);
        cells_.v4[idx] = tmp_cells_.v4[idx] + params.omega * (d_equ[4] - tmp_cells_.v4[idx]);
        cells_.v5[idx] = tmp_cells_.v5[idx] + params.omega * (d_equ[5] - tmp_cells_.v5[idx]);
        cells_.v6[idx] = tmp_cells_.v6[idx] + params.omega * (d_equ[6] - tmp_cells_.v6[idx]);
        cells_.v7[idx] = tmp_cells_.v7[idx] + params.omega * (d_equ[7] - tmp_cells_.v7[idx]);
        cells_.v8[idx] = tmp_cells_.v8[idx] + params.omega * (d_equ[8] - tmp_cells_.v8[idx]);
        tot_u += sqrtf(u_x * u_x + u_y * u_y);
      }
    }
  }
  return tot_u;
}

float av_velocity(const t_param params, t_speed_soa* tmp_cells, int* obstacles)
{
  _ASSUME_ALIGNED_SOA(tmp_cells)
  __assume_aligned(obstacles, 64);
  _ASSUME_PARAMS(params)
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int idx = ii + jj*params.nx;
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = tmp_cells->v0[idx]
                      + tmp_cells->v1[idx]
                      + tmp_cells->v2[idx]
                      + tmp_cells->v3[idx]
                      + tmp_cells->v4[idx]
                      + tmp_cells->v5[idx]
                      + tmp_cells->v6[idx]
                      + tmp_cells->v7[idx]
                      + tmp_cells->v8[idx];

        /* compute x velocity component */
        float u_x = (tmp_cells->v1[idx]
                      + tmp_cells->v5[idx]
                      + tmp_cells->v8[idx]
                      - (tmp_cells->v3[idx]
                          + tmp_cells->v6[idx]
                          + tmp_cells->v7[idx]))
                      / local_density;
        /* compute y velocity component */
        float u_y = (tmp_cells->v2[idx]
                      + tmp_cells->v5[idx]
                      + tmp_cells->v6[idx]
                      - (tmp_cells->v4[idx]
                          + tmp_cells->v7[idx]
                          + tmp_cells->v8[idx]))
                      / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed_soa** start_cells_ptr, t_speed_soa** tmp_cells_ptr, t_speed_soa** end_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int mpi_rank, int mpi_size)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /* main grid */
  // Number of cells: num columns * (numrows + 2) 
  // We need the extra two rows for the halo cells

  // Calculate inclusive start row and exclusive end row 
  int base_rowcount = params->ny / mpi_size;
  int leftover_rows = params->ny % mpi_size;
  int rowcount = (mpi_rank < leftover_rows) ? base_rowcount + 1 : base_rowcount;

  int start_row;
  if (mpi_rank < leftover_rows) {
      start_row = mpi_rank * (base_rowcount + 1);
  } else {
      start_row = (leftover_rows * (base_rowcount + 1)) + (mpi_rank - leftover_rows) * base_rowcount;
  }
  int end_row = start_row + rowcount;

  const int numcells = params->nx * (rowcount + 2);
  (*start_cells_ptr) = alloc_t_speed_soa(numcells);
  if (*start_cells_ptr == NULL) die("cannot allocate memory for start_cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  (*tmp_cells_ptr) = alloc_t_speed_soa(numcells);
  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  (*end_cells_ptr) = alloc_t_speed_soa(numcells);
  if (*end_cells_ptr == NULL) die("cannot allocate memory for end_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  (*obstacles_ptr) = (int*) _mm_malloc(sizeof(int) * numcells, 64);
  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  /* do this in parallel for numa reasons */

  _ASSUME_ALIGNED_SOA((*start_cells_ptr))
  _ASSUME_ALIGNED_SOA((*tmp_cells_ptr))
  _ASSUME_ALIGNED_SOA((*end_cells_ptr))
  _ASSUME_PARAMS((*params))

  // #pragma omp parallel for 
  for (int jj = 0; jj < rowcount + 2; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params->nx; ii++)
    {
      const int idx = ii + jj * params->nx;
      // if this is a halo cell
      if (jj == 0 || jj == rowcount + 1) {
        SOA_SET_ALL((*start_cells_ptr), idx, 0);
        SOA_SET_ALL((*tmp_cells_ptr), idx, 0);
        SOA_SET_ALL((*end_cells_ptr), idx, 0);
      } else {
        (*start_cells_ptr)->v0[idx] = w0;
        (*start_cells_ptr)->v1[idx] = w1;
        (*start_cells_ptr)->v2[idx] = w1;
        (*start_cells_ptr)->v3[idx] = w1;
        (*start_cells_ptr)->v4[idx] = w1;
        (*start_cells_ptr)->v5[idx] = w2;
        (*start_cells_ptr)->v6[idx] = w2;
        (*start_cells_ptr)->v7[idx] = w2;
        (*start_cells_ptr)->v8[idx] = w2;

        (*tmp_cells_ptr)->v0[idx] = w0;
        (*tmp_cells_ptr)->v1[idx] = w1;
        (*tmp_cells_ptr)->v2[idx] = w1;
        (*tmp_cells_ptr)->v3[idx] = w1;
        (*tmp_cells_ptr)->v4[idx] = w1;
        (*tmp_cells_ptr)->v5[idx] = w2;
        (*tmp_cells_ptr)->v6[idx] = w2;
        (*tmp_cells_ptr)->v7[idx] = w2;
        (*tmp_cells_ptr)->v8[idx] = w2;

        (*end_cells_ptr)->v0[idx] = w0;
        (*end_cells_ptr)->v1[idx] = w1;
        (*end_cells_ptr)->v2[idx] = w1;
        (*end_cells_ptr)->v3[idx] = w1;
        (*end_cells_ptr)->v4[idx] = w1;
        (*end_cells_ptr)->v5[idx] = w2;
        (*end_cells_ptr)->v6[idx] = w2;
        (*end_cells_ptr)->v7[idx] = w2;
        (*end_cells_ptr)->v8[idx] = w2;
      }
    }
  }


  /* first set all cells in obstacle array to zero */
  // #pragma omp parallel for 
  for (int jj = 0; jj < rowcount; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }


  


  /* read-in the blocked cells list */
  // We only need to read the cells that are for the current rank
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    // Skip if this cell is not for the current rank
    if (yy < start_row || yy >= end_row) continue;

    /* assign to array */
    const int row_conversion = yy - start_row;
    (*obstacles_ptr)[xx + row_conversion * params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed_soa** start_cells_ptr, t_speed_soa** tmp_cells_ptr, t_speed_soa** end_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */

  free_t_speed_soa(*start_cells_ptr);
  *start_cells_ptr = NULL;

  free_t_speed_soa(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free_t_speed_soa(*end_cells_ptr);
  *end_cells_ptr = NULL;

  _mm_free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed_soa* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}


float total_density(const t_param params, t_speed_soa* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      total += cells->v0[ii + jj*params.nx] + cells->v1[ii + jj*params.nx]
             + cells->v2[ii + jj*params.nx] + cells->v3[ii + jj*params.nx]
             + cells->v4[ii + jj*params.nx] + cells->v5[ii + jj*params.nx]
             + cells->v6[ii + jj*params.nx] + cells->v7[ii + jj*params.nx]
             + cells->v8[ii + jj*params.nx];
    }
  }

  return total;
}


int write_values(const t_param params, t_speed_soa* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        int idx = ii + jj*params.nx;
        local_density = SOA_SUM(cells, idx);

        /* compute x velocity component */
        float u_x = (cells->v1[idx]
                      + cells->v5[idx]
                      + cells->v8[idx]
                      - (cells->v3[idx]
                          + cells->v6[idx]
                          + cells->v7[idx]))
                      / local_density;
        /* compute y velocity component */
        float u_y = (cells->v2[idx]
                      + cells->v5[idx]
                      + cells->v6[idx]
                      - (cells->v4[idx]
                          + cells->v7[idx]
                          + cells->v8[idx]))
                      / local_density;

        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
