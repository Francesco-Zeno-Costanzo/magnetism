/*
Simple 2D Landau-Lifshitz-Gilbert (LLG) simulation on a square lattice
with nearest-neighbor exchange interaction and external magnetic field.
For simplicity we use an explicit Euler integration scheme.
Saves spin configurations at intervals in img folder for visualization.
Compile with: gcc simple_llg_2d.c -o llg -O3 -lm
Use plot.py to visualize results.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*===========================================================*/
/* Computational parameters                                  */
/*===========================================================*/

#define N          16    // Grid size: NxN lattice
#define NSTEPS     5000  // Total simulation steps
#define SAVE_EVERY 50    // Save every SAVE_EVERY steps
#define J          1.0   // Exchange coupling (ferromagnetic)
#define GAMMA      1.76  // Gyromagnetic ratio
#define ALPHA      0.1   // Gilbert damping (dimensionless)
#define dt         0.005 // Time step (arbitrary units)
#define Hx         0.0   // External field components
#define Hy         0.0
#define Hz         1.0


/*===========================================================*/
/* Data structure and lattice as global variable             */
/*===========================================================*/
/* Each Spin is a 3D vector*/
typedef struct {
    double x, y, z;
} vec3;

vec3 S[N][N]; // The spin lattice

/*===========================================================*/
/* Auxiliary functions                                       */
/*===========================================================*/

void normalize(vec3 *v){
    /*
    Function to normalize a 3D vector to unit length
    Patameters:
        v: pointer to vec3 structure
    */
    double n = sqrt(v->x*v->x + v->y*v->y + v->z*v->z);
    v->x /= n; v->y /= n; v->z /= n;
}

vec3 cross(vec3 a, vec3 b){
    /*
    Function to compute the cross product of two 3D vectors
    Parameters:
        a, b: vec3 structures
    Returns:
        c: vec3 structure representing the cross product a x b
    */
    vec3 c = {a.y*b.z - a.z*b.y,
              a.z*b.x - a.x*b.z,
              a.x*b.y - a.y*b.x};
    return c;
}

vec3 add(vec3 a, vec3 b){
    /*
    Function to add two 3D vectors
    Parameters:
        a, b: vec3 structures
    Returns:
        c: vec3 structure representing the sum a + b
    */
    vec3 c = {a.x + b.x, a.y + b.y, a.z + b.z};
    return c;
}

vec3 scale(vec3 a, double s){
    /*
    Function to scale a 3D vector by a scalar
    Parameters:
        a: vec3 structure
        s: double, scalar value
    Returns:
        c: vec3 structure representing the scaled vector s * a
    */
    vec3 c = {a.x * s, a.y * s, a.z * s};
    return c;
}

void save_snapshot(int step){
    /*
    Function to save the spin configuration to a CSV file
    Parameters:
        step: int, current simulation step
    */
    char fname[64];
    sprintf(fname, "img/spins_step%04d.csv", step);
    FILE *fp = fopen(fname, "w");
    if (!fp) { printf("Failed to open file %s\n", fname); exit(1);}

    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            fprintf(fp, "%d,%d,%.6f,%.6f,%.6f\n", i, j, S[i][j].x, S[i][j].y, S[i][j].z);
        }
    }

    fclose(fp);
}

/*===========================================================*/
/* Core function of simulations                              */
/*===========================================================*/

vec3 effective_field(int i, int j){
    /*
    Function to compute the effective magnetic field at site (i,j)
    Parameters:
        i, j: int, lattice indices
    Returns:
        H_eff: vec3 structure representing the effective field at (i,j)
    */
    vec3 H_eff = {Hx, Hy, Hz};

    // 4 nearest neighbors: periodic boundaries
    int ip = (i+1)%N, im = (i+N-1)%N;
    int jp = (j+1)%N, jm = (j+N-1)%N;

    // Exchange field contribution
    H_eff = add(H_eff, scale(add(add(S[ip][j], S[im][j]), add(S[i][jp], S[i][jm])), J));

    return H_eff;
}

void llg_step(){
    /*
    Function to perform a single LLG time step update on the entire lattice
    */
    
    static vec3 S_new[N][N]; // Temporary array for updates

    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){

            vec3 m         = S[i][j];
            vec3 H_eff     = effective_field(i, j);
            vec3 mxH_eff   = cross(m, H_eff);
            vec3 mxmxH_eff = cross(m, mxH_eff);

            // LLG: dm/dt = -γ m×H + α m×(m×H)
            vec3 dm = add(scale(mxH_eff, -GAMMA), scale(mxmxH_eff, -ALPHA*GAMMA));

            // Euler update
            S_new[i][j] = add(m, scale(dm, dt));
            normalize(&S_new[i][j]);
        }
    }
    
    memcpy(S, S_new, sizeof(S)); // Update the lattice
}

void init() {
    /*
    Function to initialize the spin lattice with a specific configuration
    */
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            S[i][j].x = 0.0;
            S[i][j].y = 1.0/sqrt(2.0);
            S[i][j].z = -1.0/sqrt(2.0);
        }
    }
}

/*===========================================================*/
/* Main                                                      */
/*===========================================================*/

int main(){
    init();   // Initialize the lattice
    save_snapshot(0); // Save initial configuration

    for (int step=1; step<=NSTEPS; step++){
        llg_step(); // Perform LLG update

        if (step % SAVE_EVERY == 0){
            save_snapshot(step);
        }
    }
    return 0;
}