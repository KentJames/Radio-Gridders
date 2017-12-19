#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <getopt.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <complex.h>
#include <fftw3.h>
#include <time.h>

#include "grid.h"


void image_dft(double complex *uvgrid, int grid_size, double lambda,
	       struct vis_data *vis, int iter){


    int total_steps = grid_size;
    int steps_completed = 0;
#pragma omp parallel for schedule(dynamic) 
    for (int y = 0; y<grid_size; y+=10){

        int l = (y - grid_size / 2)/lambda;

        for (int x = 0; x<grid_size; x+=iter){    
            int m = (x - grid_size / 2)/lambda;
            double real_p = 0;
            double complex_p = 0;
            for(int bl = 0; bl < vis->bl_count; ++bl){
                for(int time = 0; time < vis->bl[bl].time_count; ++time){

                    for (int freq = 0; freq < vis->bl[bl].freq_count; ++freq){
                        
                        double complex visibility = vis->bl[bl].vis[time*vis->bl[bl].freq_count + freq];

                        double subang1 = m * vis->bl[bl].uvw[time*vis->bl[bl].freq_count + freq];
                        double subang2 = l * vis->bl[bl].uvw[time*vis->bl[bl].freq_count + freq + 1];
                        double subang3 = (sqrt(1-l*l-m*m)-1) * vis->bl[bl].uvw[time*vis->bl[bl].freq_count + freq + 2];
                        
                        double angle = 2 * M_PI * subang1 + subang2 + subang3;

                        real_p += creal(visibility) * cos(angle) + cimag(visibility) * sin(angle);
                        complex_p += -creal(visibility) * sin(angle) + cimag(visibility) * cos(angle);

                    }
                }
             }
            uvgrid[y*grid_size + x] = real_p + complex_p * I;
        
        //printf("Progress: %d/%d \r",(y*grid_size + x),(grid_size*grid_size));
        }
    #pragma omp atomic
        ++steps_completed;
    #pragma omp critical
        printf("Progress: %d/%d \r",steps_completed,total_steps);
    }
}


int main(int argc, char *argv[]){


    //Structure for reporting memory usage:
    struct rusage *rusage_cp = malloc(sizeof(struct rusage));
    // Read parameters
    static struct option options[] =
      {
        {"theta",   required_argument, 0, 't' },
        {"lambda",  required_argument, 0, 'l' },
        {"image",   optional_argument, 0, 'i' },
        {"min-bl",  optional_argument, 0, 'b' },
        {"max-bl",  optional_argument, 0, 'B' },
	{"iter",    optional_argument, 0, 'I' },
        {0, 0, 0, 0}
      };
    int option_index = 0;
    double theta = 0, lambda = 0;
    char *image_file = NULL;
    double bl_min = DBL_MIN, bl_max = DBL_MAX;
    int c; int invalid = 0;
    long iter = 1;
    while ((c = getopt_long(argc, argv, ":", options, &option_index)) != -1) {
        switch(c) {
        case 't': theta = atof(optarg); break;
        case 'l': lambda = atof(optarg); break;
        case 'i': image_file = optarg; break;
        case 'b': bl_min = atof(optarg); break;
        case 'B': bl_max = atof(optarg); break;
	case 'I': iter = atol(optarg); break;
        default: invalid = 1; break;
        }
    }

    // Check grid parameters
    int grid_size = (int)(theta * lambda);
    size_t grid_byte_size = grid_size * grid_size * sizeof(double complex);
    if (grid_size <= 0) {
        fprintf(stderr, "Invalid grid configuration!\n");
        invalid = 1;
    }

    // Must have an input file
    const char *vis_file = 0;
    if (optind + 1 == argc) {
        vis_file = argv[optind];
    } else {
        printf("Please supply a visibility input file!\n");
        invalid = 1;
    }
    if (invalid) {
        printf("usage: %s --theta=THETA --lambda=LAM [--image=IMAGE]\n", argv[0]);
        printf("              [--min-bl=MIN_BL] [--max-bl=MAX_BL]\n");
        printf("              INPUT\n");
        printf("\n");
        printf("optional arguments:\n");
        printf("  --theta=THETA         Field of view size (in radians)\n");
        printf("  --lambda=LAM          uv grid size (in wavelenghts)\n");
        printf("  --image=IMAGE         image output file\n");
        printf("  --min-bl=MIN_BL       Minimum baseline length to consider (in km)\n");
        printf("  --max-bl=MAX_BL       Maximum baseline length to consider (in km)\n");
	printf("  --iter=ITER           Samples every +=ITER point in fourier space. Quickens DFT.\n");
        printf("positional arguments:\n");
        printf("  input                 input visibilities\n");
        return 1;
    }

    // Intialise HDF5
    init_dtype_cpx();

    // Open files
    struct vis_data vis;
    int grid_fd = -1, image_fd = -1;
    if (load_vis(vis_file, &vis, bl_min, bl_max)) {
        return 1;
    }

    if (image_file) {
        image_fd = open(image_file, O_CREAT | O_TRUNC | O_WRONLY, 0666);
        if (image_fd == -1) {
            perror("Failed to open image file");
            return 1;
        }
    }
    // Allocate grid
    printf("\nGrid size:    %d x %d (%.2f GB)\n", grid_size, grid_size, (double)(grid_byte_size)/1000000000);
    double complex *uvgrid = (double complex *)calloc(grid_byte_size, 1);

    // Simple uniform weight (we re-use the grid to save an allocation)
    printf("Weighting...\n");
    weight((unsigned int *)uvgrid, grid_size, theta, &vis);
    memset(uvgrid, 0, grid_size * grid_size * sizeof(unsigned int));

    // Set up performance counters
    struct perf_counters counters;
    open_perf_counters(&counters);

    // Start timer
    struct timespec start_time;
    clock_gettime(CLOCK_REALTIME, &start_time);

    uint64_t flops = 0, mem = 0;
    printf("Direct DFT...(this takes a LONG time)\n");

    if(iter>1) printf("Sampling every %ld points in fourier space (saves time).\n",iter);
    
    // DFT HERE
    image_dft(uvgrid, grid_size, lambda, &vis,iter); 
    struct timespec end_time;
    clock_gettime(CLOCK_REALTIME, &end_time);
    printf("\nGrid-Time:  %.3f",
           (double)(end_time.tv_sec - start_time.tv_sec) +
           (double)(end_time.tv_nsec - start_time.tv_nsec) / 1000000000);

    //Lets get some memory stats:
    getrusage(RUSAGE_SELF, rusage_cp);
    printf("\nMaximum Grid Memory: %.2f GB", (float)rusage_cp->ru_maxrss/(1024*1024));

    // Show performance counters after gridding
    printf("\nCounters:\n");
    print_perf_counters(&counters, flops, mem);

    // Make hermitian
    printf("\nMake hermitian...\n");
    make_hermitian(uvgrid, grid_size);


    if (image_fd != -1) {
        printf("FFT...\n");

        // First shift zero frequency
        fft_shift(uvgrid, grid_size);

        // Do DFT. Complex-to-complex to keep with numpy (TODO: optimize)
        fftw_plan plan;
        plan = fftw_plan_dft_2d(grid_size, grid_size, uvgrid, uvgrid, -1, FFTW_ESTIMATE);
        fftw_execute_dft(plan, uvgrid, uvgrid);

        // Shift zero frequency back into centre
        fft_shift(uvgrid, grid_size);

        // Write real part to disk
        printf("Write image...\n");
        int i;
        double *row = malloc(sizeof(double) * grid_size);
        for (i = 0; i < grid_size; i++) {
            int j;
            for (j = 0; j < grid_size; j++) {
                row[j] = creal(uvgrid[i*grid_size+j]);
            }
            write(image_fd, row, sizeof(double) * grid_size);
        }
        close(image_fd);
    }
    getrusage(RUSAGE_SELF, rusage_cp);
    printf("\nMax Memory: %.2f GB", (float)rusage_cp->ru_maxrss/(1024*1024));


    return 0;
}
