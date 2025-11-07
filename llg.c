/*
Simple 2D Landau-Lifshitz-Gilbert (LLG) simulation on a square lattice
To compute the effective field we considerer the exchange interaction,
the Dzyaloshinskii–Moriya interaction, uniaxial anisotropy and an
external applied field.

Several initial configuration are implemented and the integration
in done with the Heun predictor and corrector alghoritm.
Available initial configurations:
        | "random"     -> fully random spins
       /  "vortex"     -> vortex-like pattern
mode <    "quadrants"  -> four-domain configuration (+-z)
       \  "spiral"     -> spin spiral along x
        | "bubble"     -> single reversed domain in the center

Saves spin configurations at intervals in img folder for visualization.

Compile with: gcc llg.c -o llg -O3 -lm
run with    : ./llg mode

Use plot.py to visualize results.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*===========================================================*/
/* Computational parameters                                  */
/*===========================================================*/

#define N          64     // Grid size: NxN lattice
#define NSTEPS     10000  // Total simulation steps
#define SAVE_EVERY 50     // Save every SAVE_EVERY steps
#define J          1      // Exchange coupling (ferromagnetic)
#define GAMMA      1.76   // Gyromagnetic ratio
#define ALPHA      0.1    // Gilbert damping (dimensionless)
#define KU         1      // Anisotropy constant
#define D          1      // DMI coupling constant
#define dt         0.005  // Time step (arbitrary units)
#define Hx         0.0    // External field components
#define Hy         0.0
#define Hz         0.25

/*===========================================================*/
/* Data structure and lattice as global variable             */
/*===========================================================*/
/* Each Spin is a 3D vector*/
typedef struct {
    double x, y, z;
} vec3;


static vec3 *m;            /* Current spins (size N*N) */
static vec3 *m_pred;       /* Predicted spins for Heun */
static vec3 *Heff_orig;    /* Effective field at original spins  */
static vec3 *Heff_pred;    /* Effective field at predicted spins */

/* Anisotropy axis as vec3 */
static vec3 K_u = {0.0, 1.0, 0.0};

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

double dot(vec3 a, vec3 b){
    /*
    Function to comput dot product of two 3D vectors
    Parameters:
        a, b: vec3 structures
    Returns:
        double representing the dot product a*b
    */
    return a.x*b.x + a.y*b.y + a.z*b.z;
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

int idx(int i, int j){
    /*
    Function to map a point in a matrix
    in a point in an array
    Parameters:
        i, j: int, point coordinates
    Return:
        index in the array
    */
    return i + j*N;
}

void save_snapshot(int step){
     /*
    Function to save the spin configuration to a CSV file
    Parameters:
        step: int, current simulation step
    */
    char fname[512];
    
    sprintf(fname, "img/spins_step%06d.csv", step);
    FILE *f = fopen(fname, "w");
    if(!f){ fprintf(stderr, "Cannot open %s for writing", fname); return; }
    
    for(int j=0; j<N; j++){
        for(int i=0; i<N; i++){
            int id = idx(i,j);
            fprintf(f, "%d,%d,%.8f,%.8f,%.8f\n", i, j, m[id].x, m[id].y, m[id].z);
        }
    }
    fclose(f);
}

/*===========================================================*/
/* Core function of simulations                              */
/*===========================================================*/

void effective_field(vec3 *m_arr, vec3 *Hout){
    /*
    Function to compute the effective magnetic field
    Parameters:
        m_arr: pointer to vec3 structure array
            Lattice of the system
        Hout: pointer to vec3 structure array
            Effective Field
    */
    
    for(int j=0; j<N; j++){
        for(int i=0; i<N; i++){

            int id = idx(i,j);
            
            /* Initialize the filed with the external applied one */
            vec3 H_ext = {Hx, Hy, Hz};

            /* Exchange: nearest neighbours (discrete Laplacian) */
            int im = (i - 1 + N) % N;
            int ip = (i + 1)     % N;
            int jm = (j - 1 + N) % N;
            int jp = (j + 1)     % N;

            vec3 m_ip = m_arr[idx(ip, j)];
            vec3 m_im = m_arr[idx(im, j)];
            vec3 m_jp = m_arr[idx(i, jp)];
            vec3 m_jm = m_arr[idx(i, jm)];
            vec3 m0   = m_arr[id];

            vec3 sum_neigh = {0.0, 0.0, 0.0};
            sum_neigh = add(sum_neigh, m_ip);
            sum_neigh = add(sum_neigh, m_im);
            sum_neigh = add(sum_neigh, m_jp);
            sum_neigh = add(sum_neigh, m_jm);

            vec3 laplacian = add(sum_neigh, scale(m0, -4.0));
            vec3 H_exch    = scale(laplacian, J);

            // DMI interaction
            vec3 H_dmi = {0.0, 0.0, 0.0};

            H_dmi.x = -D * (m_jp.z - m_jm.z);
            H_dmi.y =  D * (m_ip.z - m_im.z);
            H_dmi.z =  D * ((m_jp.x - m_jm.x) - (m_ip.y - m_im.y));

            // anisotropy term
            vec3 m      = m_arr[id];
            double proj = dot(m, K_u);
            vec3 H_anis = scale(K_u, 2*proj*KU);

            Hout[id] = add(H_ext, add(H_dmi, add(H_exch, H_anis)));
        }
    }
}


void llg_rhs(const vec3 *mi, const vec3 *h, vec3 *rhs){
    /*
    Function to compute the right hand side of llg equation
    dm/dt = -GAMMA/(1+ALPHA^2) [ m x H + ALPHA * m x (m x H) ]

    Parameters:
        m_i: pointer to vec3 structure array
            Lattice of the system
        h: pointer to vec3 structure array
            Effective Field
        rhs: pointer to vec3 structure array
            Equation to integrate
     
    */
    vec3 mxh    = cross(*mi, *h);
    vec3 mxmxh  = cross(*mi, mxh);
    double pref = -GAMMA / (1.0 + ALPHA * ALPHA);
    vec3 term   = add(mxh, scale(mxmxh, ALPHA));

    *rhs        = scale(term, pref);
}


void llg_step_heun(){
    /*
    Function to perform a single LLG time step update on the entire lattice 
    */

    /* 1) compute Heff at original spins */
    effective_field(m, Heff_orig);

    /* 2) predictor: m_pred = m + dt * f(m, Heff_orig) */
    for(int i=0; i<N*N; i++){
        vec3 dm;

        llg_rhs(&m[i], &Heff_orig[i], &dm);
        
        m_pred[i].x = m[i].x + dt * dm.x;
        m_pred[i].y = m[i].y + dt * dm.y;
        m_pred[i].z = m[i].z + dt * dm.z;
        
        normalize(&m_pred[i]);
    }

    /* 3) compute Heff at predicted spins */
    effective_field(m_pred, Heff_pred);

    /* 4) corrector: m_new = m + 0.5*dt*( f(m,Heff_orig) + f(m_pred,Heff_pred) ) */
    for(int i=0; i<N*N; i++){
        vec3 dm_o, dm_p;

        llg_rhs(&m[i],      &Heff_orig[i], &dm_o);
        llg_rhs(&m_pred[i], &Heff_pred[i], &dm_p);

        m[i].x = m[i].x + 0.5 * dt * (dm_o.x + dm_p.x);
        m[i].y = m[i].y + 0.5 * dt * (dm_o.y + dm_p.y);
        m[i].z = m[i].z + 0.5 * dt * (dm_o.z + dm_p.z);

        normalize(&m[i]);
    }
}

void init(const char *mode){
    /*
    Function to initialize the spin lattice with a specific configuration
    Available modes:
        "random"     -> fully random spins
        "vortex"     -> vortex-like pattern
        "quadrants"  -> four-domain configuration (+-z)
        "spiral"     -> spin spiral along x
        "bubble"     -> single reversed domain in the center
    */
    
    srand((unsigned)time(NULL));

    // Center of the grid
    double cx = 0.5 * N;
    double cy = 0.5 * N;
    
    if (strcmp(mode, "random") == 0){
        for(int j=0; j<N; j++){
            for(int i=0; i<N; i++){
                
                int id = idx(i,j);
                
                m[id].x = 2*((rand() / (double)RAND_MAX) - 0.5);
                m[id].y = 2*((rand() / (double)RAND_MAX) - 0.5);
                m[id].z = 2*((rand() / (double)RAND_MAX) - 0.5);
                normalize(&m[id]);
            }
        }
    }
    
    else if (strcmp(mode, "vortex") == 0){
        for(int j=0; j<N; j++){
            for(int i=0; i<N; i++){
                int id = idx(i, j);
                double dx = i - cx;
                double dy = j - cy;
                double r = sqrt(dx*dx + dy*dy) + 1e-9;
                m[id].x = -dx / r;
                m[id].y =  dy / r;
                m[id].z = 1 * ((r < N/3) ? 1 : -1);  // core up/down
                normalize(&m[id]);
            }
        }
    }
    
    else if(strcmp(mode, "quadrants") == 0){
        double width = N / 10.0;   // controls smoothness of domain wall

        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                int id = idx(i, j);

                double sx = (i - cx) / width;
                double sy = (j - cy) / width;

                // smooth sign transitions across the midlines using tanh
                double sign_x = tanh(sx);
                double sign_y = tanh(sy);

                // combine signs to get quadrant pattern
                double mz = -sign_x * sign_y;

                // create a small in-plane rotation for a smoother edge
                double angle = atan2(j - cy, i - cx);
                m[id].x = 0.2 * cos(angle);
                m[id].y = 0.2 * sin(angle);
                m[id].z = mz;

                normalize(&m[id]);
            }
        }
    }
    
    else if(strcmp(mode, "spiral") == 0){
        for(int j=0; j<N; j++){
            for(int i=0; i<N; i++){
                int id = idx(i, j);
                double phi = 2.0 * 4*atan(1) * i / N;
                m[id].x = cos(phi);
                m[id].y = sin(phi);
                m[id].z = 0.0;
            }
        }
    }

    else if(strcmp(mode, "bubble") == 0){
        double R  = N / 4.0;

        for(int j=0; j<N; j++){
            for(int i=0; i<N; i++){

                int id = idx(i, j);
                double dx = i - cx;
                double dy = j - cy;
                double r2 = dx*dx + dy*dy;

                if(r2 < R*R){
                    m[id].x = 0.0;
                    m[id].y = 0.0;
                    m[id].z = -1.0;  // reversed domain
                } 
                else{
                    m[id].x = 0.0;
                    m[id].y = 0.0;
                    m[id].z = 1.0;
                }
            }
        }
    }
    else{
        fprintf(stderr, "Unknown init mode '%s'.\n", mode);
        exit(0);
    }
}

/*===========================================================*/
/* Main                                                      */
/*===========================================================*/

int main(int argc, char *argv[]){

    const char *mode = "random";
    if(argc > 1){
        mode = argv[1];
    }
    
    m         = (vec3*)malloc(sizeof(vec3)*N*N);
    m_pred    = (vec3*)malloc(sizeof(vec3)*N*N);
    Heff_orig = (vec3*)malloc(sizeof(vec3)*N*N);
    Heff_pred = (vec3*)malloc(sizeof(vec3)*N*N);

    if(!m || !m_pred || !Heff_orig || !Heff_pred){
        fprintf(stderr, "alloc fail");
        return 1;
    }

    init(mode);

    for(int step=0; step<=NSTEPS; step++){
        llg_step_heun();

        if(step % SAVE_EVERY == 0){
            save_snapshot(step);
        }
        
        int barWidth    = 40;
        double progress = (double)step / NSTEPS;
        printf("\r[");

        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) printf("=");
            else if (i == pos) printf(">");
            else printf(" ");
        }
        printf("] %.1f%%", progress * 100.0);
        fflush(stdout);
    }
    printf("\n");

    free(m);
    free(m_pred);
    free(Heff_orig);
    free(Heff_pred);
    return 0;
}