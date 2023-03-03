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
  float* v0;
  float* v1;
  float* v2;
  float* v3;
  float* v4;
  float* v5;
  float* v6;
  float* v7;
  float* v8;
} t_speed_soa;



/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/* Allocate memory for a t_speed_soa structure */
t_speed_soa* alloc_t_speed_soa(const int size)
{
  t_speed_soa* speeds = (t_speed_soa*) _mm_malloc(sizeof(t_speed_soa), 64);
  speeds->v0 = (float*) _mm_malloc(sizeof(float) * size, 64);
  speeds->v1 = (float*) _mm_malloc(sizeof(float) * size, 64);
  speeds->v2 = (float*) _mm_malloc(sizeof(float) * size, 64);
  speeds->v3 = (float*) _mm_malloc(sizeof(float) * size, 64);
  speeds->v4 = (float*) _mm_malloc(sizeof(float) * size, 64);
  speeds->v5 = (float*) _mm_malloc(sizeof(float) * size, 64);
  speeds->v6 = (float*) _mm_malloc(sizeof(float) * size, 64);
  speeds->v7 = (float*) _mm_malloc(sizeof(float) * size, 64);
  speeds->v8 = (float*) _mm_malloc(sizeof(float) * size, 64);

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



/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed_soa** start_cells_ptr, t_speed_soa** tmp_cells_ptr, t_speed_soa** end_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speed_soa*restrict start_cells, t_speed_soa*restrict tmp_cells, t_speed_soa*restrict end_cells, int*restrict obstacles);
int accelerate_flow(const t_param params, t_speed_soa*restrict cells, int*restrict obstacles);
int propagate(const t_param params, t_speed_soa*restrict cells, t_speed_soa*restrict tmp_cells);
float collision(const t_param params, t_speed_soa*restrict cells, t_speed_soa*restrict tmp_cells, int*restrict obstacles);
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
  initialise(paramfile, obstaclefile, &params, &start_cells, &tmp_cells, &end_cells, &obstacles, &av_vels);

  float num_cells_without_obstacles = 0;
  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      int cell_idx = ii * params.nx + jj;
      if (!obstacles[cell_idx])
      {
        num_cells_without_obstacles++;
      }
    }
  }



  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    av_vels[tt] = timestep(params, start_cells, tmp_cells, end_cells, obstacles) 
                  / num_cells_without_obstacles;

    // swap pointers
    t_speed_soa* tmp = start_cells;
    start_cells = end_cells;
    end_cells = tmp;

#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, start_cells, obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(params, start_cells, obstacles, av_vels);
  finalise(&params, &start_cells, &tmp_cells, &end_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

float timestep(const t_param params, t_speed_soa*restrict start_cells, t_speed_soa*restrict tmp_cells, t_speed_soa*restrict end_cells, int*restrict obstacles)
{
  accelerate_flow(params, start_cells, obstacles);

  float tot_vel = 0.0f;
  /* loop over _all_ cells */
  propagate(params, start_cells, tmp_cells);
  tot_vel = collision(params, end_cells, tmp_cells, obstacles);
  return tot_vel;
}

int accelerate_flow(const t_param params, t_speed_soa*restrict cells, int*restrict obstacles)
{
  _ASSUME_ALIGNED_SOA(cells)
  __assume_aligned(obstacles, 64);
  
  __assume((params.nx) % 16 == 0);
  __assume((params.ny) % 16 == 0);
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;

  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells->v3[ii + jj*params.nx] - w1) > 0.f
        && (cells->v6[ii + jj*params.nx] - w2) > 0.f
        && (cells->v7[ii + jj*params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->v1[ii + jj*params.nx] += w1;
      cells->v5[ii + jj*params.nx] += w2;
      cells->v8[ii + jj*params.nx] += w2;
      /* decrease 'west-side' densities */
      cells->v3[ii + jj*params.nx] -= w1;
      cells->v6[ii + jj*params.nx] -= w2;
      cells->v7[ii + jj*params.nx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

int propagate(const t_param params, t_speed_soa*restrict cells, t_speed_soa*restrict tmp_cells)
{
  _ASSUME_ALIGNED_SOA(cells)
  _ASSUME_ALIGNED_SOA(tmp_cells)
  
  __assume((params.nx) % 16 == 0);
  __assume((params.ny) % 16 == 0);
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    { 
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

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


float collision(const t_param params, t_speed_soa*restrict cells, t_speed_soa*restrict tmp_cells, int*restrict obstacles)
{
  _ASSUME_ALIGNED_SOA(cells)
  _ASSUME_ALIGNED_SOA(tmp_cells)
  __assume_aligned(obstacles, 64);
  
  __assume((params.nx) % 16 == 0);
  __assume((params.ny) % 16 == 0);

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  float tot_u = 0.f; 

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    { 
      const int idx = ii + jj*params.nx;
      /* don't consider occupied cells */
      if (obstacles[idx])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        cells->v1[idx] = tmp_cells->v3[idx];
        cells->v2[idx] = tmp_cells->v4[idx];
        cells->v3[idx] = tmp_cells->v1[idx];
        cells->v4[idx] = tmp_cells->v2[idx];
        cells->v5[idx] = tmp_cells->v7[idx];
        cells->v6[idx] = tmp_cells->v8[idx];
        cells->v7[idx] = tmp_cells->v5[idx];
        cells->v8[idx] = tmp_cells->v6[idx];
      } else {
        /* compute local density total */
          /* compute local density total */
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
        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

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
                                          + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                          + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                          + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                          + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                          + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                          + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                          + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                          + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));

        /* relaxation step */
        cells->v0[idx] = tmp_cells->v0[idx] + params.omega * (d_equ[0] - tmp_cells->v0[idx]);
        cells->v1[idx] = tmp_cells->v1[idx] + params.omega * (d_equ[1] - tmp_cells->v1[idx]);
        cells->v2[idx] = tmp_cells->v2[idx] + params.omega * (d_equ[2] - tmp_cells->v2[idx]);
        cells->v3[idx] = tmp_cells->v3[idx] + params.omega * (d_equ[3] - tmp_cells->v3[idx]);
        cells->v4[idx] = tmp_cells->v4[idx] + params.omega * (d_equ[4] - tmp_cells->v4[idx]);
        cells->v5[idx] = tmp_cells->v5[idx] + params.omega * (d_equ[5] - tmp_cells->v5[idx]);
        cells->v6[idx] = tmp_cells->v6[idx] + params.omega * (d_equ[6] - tmp_cells->v6[idx]);
        cells->v7[idx] = tmp_cells->v7[idx] + params.omega * (d_equ[7] - tmp_cells->v7[idx]);
        cells->v8[idx] = tmp_cells->v8[idx] + params.omega * (d_equ[8] - tmp_cells->v8[idx]);
        tot_u += sqrtf(u_sq);
      }
    }
  }
  return tot_u;
}

float av_velocity(const t_param params, t_speed_soa* tmp_cells, int* obstacles)
{
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
               int** obstacles_ptr, float** av_vels_ptr)
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

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */



  (*start_cells_ptr) = alloc_t_speed_soa(params->ny * params->nx);
  if (*start_cells_ptr == NULL) die("cannot allocate memory for start_cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  (*tmp_cells_ptr) = alloc_t_speed_soa(params->ny * params->nx);
  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  (*end_cells_ptr) = alloc_t_speed_soa(params->ny * params->nx);
  if (*end_cells_ptr == NULL) die("cannot allocate memory for end_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  (*obstacles_ptr) = (int*) _mm_malloc(sizeof(int) * params->ny * params->nx, 64);
  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      const int idx = ii + jj * params->nx;
      (*start_cells_ptr)->v0[idx] = w0;
      (*start_cells_ptr)->v1[idx] = w1;
      (*start_cells_ptr)->v2[idx] = w1;
      (*start_cells_ptr)->v3[idx] = w1;
      (*start_cells_ptr)->v4[idx] = w1;
      (*start_cells_ptr)->v5[idx] = w2;
      (*start_cells_ptr)->v6[idx] = w2;
      (*start_cells_ptr)->v7[idx] = w2;
      (*start_cells_ptr)->v8[idx] = w2;
    }
  }


  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
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
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
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

#define SOA_SUM(soa, idx) (soa->v0[idx] + soa->v1[idx] + soa->v2[idx] + soa->v3[idx] + soa->v4[idx] + soa->v5[idx] + soa->v6[idx] + soa->v7[idx] + soa->v8[idx])


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
