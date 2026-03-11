/*
Simple 2D Landau-Lifshitz-Gilbert (LLG) simulation on a square lattice
To compute the effective field we considerer the exchange interaction,
the Dzyaloshinskii–Moriya interaction, uniaxial anisotropy and an
external applied field.

dm/dt = -gamma/(1 + alpha^2) [ m × H_eff + alpha m × (m × H_eff) ]

where H_eff = -δE/δm is the effective magnetic field derived from
the micromagnetic energy functional.

The total energy functional used in this code is:

E = ∫ [ 
        J |∇m|^2
      + D ( m_z (∇·m) − (m · ∇) m_z )
      − K_u (m · k)^2
      − H_ext · m
    ] dA


Terms:
1) Exchange interaction:
    E_ex = J |∇m|^2
2) Interfacial Dzyaloshinskii–Moriya interaction (DMI):
    E_DMI = D [ m_z (∇·m) − (m · ∇) m_z ]
3) Uniaxial anisotropy:
    E_anis = −K_u (m · k)^2
4) Zeeman interaction with external field:
    E_Z = − H_ext · m

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
#include <unistd.h>
#include <getopt.h>
#include <sys/stat.h>
#include <sys/types.h>


/*===========================================================*/
/* Data structure for spins, latice and parameters           */
/*===========================================================*/

/****************************/
/* Each Spin is a 3D vector */
/****************************/
typedef struct {
    double x, y, z;
} vec3;

/***************************/
/* Computational parameters*/
/***************************/
typedef struct{
    int N;                // Grid size: NxN lattice
    int nsteps;           // Total simulation steps
    int save_every;       // Save every steps
    double J;             // Exchange coupling (ferromagnetic)
    double gamma;         // Gyromagnetic ratio
    double alpha;         // Gilbert damping (dimensionless)
    double K_u;           // Anisotropy constant
    double D;             // DMI coupling constant
    double dt;            // Time step (arbitrary units)
    double Hx, Hy, Hz;    // External field components
    double kx, ky, kz;    // Anisotropy directions
} Params;

/****************************/
/* State of the simulation  */
/****************************/
typedef struct{
    int N; 

    vec3 *m;             /* Current spins (size N*N) */
    vec3 *m_pred;        /* Predicted spins for Heun */
    vec3 *Heff_orig;     /* Effective field at original spins */
    vec3 *Heff_pred;     /* Effective field at predicted spins */

    vec3 K_u_dir;        /* Anisotropy axis as vec3 */

    const Params *params; 

} SimulationState;

void default_params(Params *p)
{
    p->N          = 64;
    p->nsteps     = 10000;
    p->save_every = 50;
    p->J          = 1.0;
    p->gamma      = 1.76;
    p->alpha      = 0.1;
    p->K_u        = 1.0;
    p->D          = 1.0;
    p->dt         = 0.005;
    p->Hx         = 0.0;
    p->Hy         = 0.0;
    p->Hz         = 0.25;
    p->kx         = 0.0;
    p->ky         = 1.0;
    p->kz         = 0.0;
}

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
    if(n > 1e-12){
        v->x /= n;
        v->y /= n;
        v->z /= n;
    }
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

int idx(int i, int j, int N){
    /*
    Function to map a point in a matrix
    in a point in an array
    Parameters:
        i, j: int, point coordinates
        N : int, length of the lattice, N**2 is the area. 
    Return:
        index in the array
    */
    return i + j*N;
}

void save_snapshot(SimulationState *state, int step, const char *outdir){
     /*
    Function to save the spin configuration to a CSV file
    Parameters:
        state: pointer to SimulationState structure
        step: int, current simulation step
        outdir: const char *, output directory path
    */
    char fname[512];
    int N = state->N;
    
    sprintf(fname, "%s/spins_step%06d.csv", outdir, step);
    FILE *f = fopen(fname, "w");
    if(!f){ fprintf(stderr, "Cannot open %s for writing", fname); return; }
    
    for(int j=0; j<N; j++){
        for(int i=0; i<N; i++){
            int id = idx(i,j, N);
            fprintf(f, "%d,%d,%.8f,%.8f,%.8f\n", 
                i, j, state->m[id].x, state->m[id].y, state->m[id].z);
        }
    }
    fclose(f);
}

void save_params(const Params *p, const char *mode, const char *outdir){
    /*
    Function to save the simulation parameters to a text file for record keeping
    Parameters:
        p: pointer to Params structure containing simulation parameters
        mode: const char *, initial configuration mode
        outdir: const char *, output directory path
    */
    char fname[512];
    sprintf(fname, "%s/params.txt", outdir);

    FILE *f = fopen(fname, "w");
    if(!f){
        printf("Cannot write params file\n");
        return;
    }

    fprintf(f, "Simulation parameters\n");
    fprintf(f, "---------------------\n");

    fprintf(f, "N          %d\n", p->N);
    fprintf(f, "nsteps     %d\n", p->nsteps);
    fprintf(f, "save_every %d\n", p->save_every);

    fprintf(f, "\nPhysical parameters\n");

    fprintf(f, "J      %.6f\n", p->J);
    fprintf(f, "D      %.6f\n", p->D);
    fprintf(f, "Ku     %.6f\n", p->K_u);

    fprintf(f, "\nField\n");

    fprintf(f, "Hx %.6f\n", p->Hx);
    fprintf(f, "Hy %.6f\n", p->Hy);
    fprintf(f, "Hz %.6f\n", p->Hz);

    fprintf(f, "\nDirection of Uniaxial anisotropy\n");

    fprintf(f, "kx %.6f\n", p->kx);
    fprintf(f, "ky %.6f\n", p->ky);
    fprintf(f, "kz %.6f\n", p->kz);

    fprintf(f, "\nDynamics\n");

    fprintf(f, "gamma %.6f\n", p->gamma);
    fprintf(f, "alpha %.6f\n", p->alpha);
    fprintf(f, "dt    %.6f\n", p->dt);

    fprintf(f, "\nInitial condition\n");

    fprintf(f, "mode %s\n", mode);

    fclose(f);
}

void print_help(Params *p){
    /* Help message */
    printf("Options:\n");

    printf(" -N  grid size (default %d)\n", p->N);
    printf(" -t  number of time steps (default %d)\n", p->nsteps);
    printf(" -s  save every s steps (default %d)\n", p->save_every);
    printf(" -d  timestep dt (default %f)\n", p->dt);

    printf("\nPhysical parameters\n");

    printf(" -J  exchange constant (default %f)\n", p->J);
    printf(" -D  DMI strength (default %f)\n", p->D);
    printf(" -K  anisotropy constant (default %f)\n", p->K_u);

    printf("\nExternal field\n");

    printf(" -x  Hx (default %f)\n", p->Hx);
    printf(" -y  Hy (default %f)\n", p->Hy);
    printf(" -z  Hz (default %f)\n", p->Hz);

    printf("\nAnisotropy axis\n");

    printf(" -X  kx (default %f)\n", p->kx);
    printf(" -Y  ky (default %f)\n", p->ky);
    printf(" -Z  kz (default %f)\n", p->kz);

    printf("\nSimulation\n");

    printf(" -m  initial condition (random, uniform, vortex...)\n");

    printf(" -o  output directory for saving snapshots.csv (default run0) \n");
    printf(" -e  energy file will be inside the output directory (default ene.csv) \n");

    printf("\nExample:\n");

    printf("./llg -N 128 -J 1 -D 1 -z 0.3 -m random -o run1\n\n");
}

/*===========================================================*/
/* Allocation ad deallocation of the memory                  */
/*===========================================================*/

void state_free(SimulationState *state){
    /*
    Function to free the memory allocated for the simulation state
    Parameters:
        state: pointer to SimulationState structure to free
    */
    if (!state) return;
    free(state->m);
    free(state->m_pred);
    free(state->Heff_orig);
    free(state->Heff_pred);
    free(state);
}

SimulationState* sim_state_alloc(const Params *params){
    /*
    Function to allocate memory for the simulation state and initialize it with parameters
    
    Parameters:
        params: pointer to Params structure containing simulation parameters
    
    Returns:
        pointer to allocated SimulationState structure, or NULL on failure
    */
    SimulationState *state = (SimulationState*)malloc(sizeof(SimulationState));

    if (!state){
        fprintf(stderr, "Cannot allocate simulation state\n");
        return NULL;
    }

    state->N       = params->N;
    state->params  = params;
    state->K_u_dir = (vec3){params->kx, params->ky, params->kz};
    normalize(&state->K_u_dir);

    
    int size         = params->N * params->N;
    state->m         = (vec3*)malloc(sizeof(vec3) * size);
    state->m_pred    = (vec3*)malloc(sizeof(vec3) * size);
    state->Heff_orig = (vec3*)malloc(sizeof(vec3) * size);
    state->Heff_pred = (vec3*)malloc(sizeof(vec3) * size);
    
    if (!state->m || !state->m_pred || !state->Heff_orig || !state->Heff_pred) {
        fprintf(stderr, "Cannot allocate spin arrays\n");
        state_free(state);
        return NULL;
    }
    
    return state;

}

/*===========================================================*/
/* Core function of simulations                              */
/*===========================================================*/

void effective_field(SimulationState *state, vec3 *m_arr, vec3 *Hout){
    /*
    Function to compute the effective magnetic field
    Parameters:
        state: pointer to SimulationState structure
        m_arr: pointer to vec3 structure array
            Lattice of the system
        Hout: pointer to vec3 structure array
            Effective Field
    */
    
    int N = state->N;
    const Params *p = state->params;
    
    for(int j=0; j<N; j++){
        for(int i=0; i<N; i++){

            int id = idx(i, j, N);
            
            /* Initialize the filed with the external applied one */
            vec3 H_ext = {p->Hx, p->Hy, p->Hz};

            /* Exchange: nearest neighbours (discrete Laplacian) */
            int im = (i - 1 + N) % N;
            int ip = (i + 1)     % N;
            int jm = (j - 1 + N) % N;
            int jp = (j + 1)     % N;

            vec3 m_ip = m_arr[idx(ip, j, N)];
            vec3 m_im = m_arr[idx(im, j, N)];
            vec3 m_jp = m_arr[idx(i, jp, N)];
            vec3 m_jm = m_arr[idx(i, jm, N)];
            vec3 m0   = m_arr[id];

            vec3 sum_neigh = {0.0, 0.0, 0.0};
            sum_neigh = add(sum_neigh, m_ip);
            sum_neigh = add(sum_neigh, m_im);
            sum_neigh = add(sum_neigh, m_jp);
            sum_neigh = add(sum_neigh, m_jm);

            vec3 laplacian = add(sum_neigh, scale(m0, -4.0));
            vec3 H_exch    = scale(laplacian, p->J);

            // DMI interaction
            vec3 H_dmi = {0.0, 0.0, 0.0};

            H_dmi.x = -p->D * (m_jp.z - m_jm.z)/2.0;
            H_dmi.y =  p->D * (m_ip.z - m_im.z)/2.0;
            H_dmi.z =  p->D * ((m_jp.x - m_jm.x) - (m_ip.y - m_im.y))/2.0;

            // anisotropy term
            vec3 m      = m_arr[id];
            double proj = dot(m, state->K_u_dir);
            vec3 H_anis = scale(state->K_u_dir, 2*proj*p->K_u);

            Hout[id] = add(H_ext, add(H_dmi, add(H_exch, H_anis)));
        }
    }
}


void llg_rhs(SimulationState *state, const vec3 *mi, const vec3 *h, vec3 *rhs){
    /*
    Function to compute the right hand side of llg equation
    dm/dt = -GAMMA/(1+ALPHA^2) [ m x H + ALPHA * m x (m x H) ]

    Parameters:
        state: pointer to SimulationState structure
        m_i: pointer to vec3 structure array
            Lattice of the system
        h: pointer to vec3 structure array
            Effective Field
        rhs: pointer to vec3 structure array
            Equation to integrate
     
    */
    const Params *p = state->params;
    
    vec3 mxh    = cross(*mi, *h);
    vec3 mxmxh  = cross(*mi, mxh);
    double pref = -p->gamma / (1.0 + p->alpha * p->alpha);
    vec3 term   = add(mxh, scale(mxmxh, p->alpha));

    *rhs        = scale(term, pref);
}

vec3 rodrigues_rotation(vec3 m, vec3 ang_vel, double dt){
    /*
    Function to perform a rotation of the spin m around the axis defined
    by the angular velocity vector ang_vel using Rodrigues' rotation formula.

    Parameters:
        m: vec3 structure, the spin to rotate
        ang_vel: vec3 structure, the angular velocity vector (axis and speed)
        dt: double, time step for the rotation
    
    Returns:
        m_new: vec3 structure, the rotated spin
    */
    double w = sqrt(dot(ang_vel, ang_vel));
    if(w < 1e-12) return m;

    vec3 k = scale(ang_vel, 1.0/w);
    double theta = w * dt;

    vec3 term1 = scale(m, cos(theta));
    vec3 term2 = scale(cross(k,m), sin(theta));
    vec3 term3 = scale(k, dot(k,m)*(1-cos(theta)));

    return add(term1, add(term2, term3));
}

void llg_step_heun(SimulationState *state){
    /*
    Function to perform a single LLG time step update on the entire lattice 
    */
    int N           = state->N;
    vec3 *m         = state->m;
    vec3 *m_pred    = state->m_pred;
    vec3 *Heff_orig = state->Heff_orig;
    vec3 *Heff_pred = state->Heff_pred;
    double dt       = state->params->dt;    

    /* 1) compute Heff at original spins */
    effective_field(state, m, Heff_orig);

    /* 2) predictor: m_pred = m + dt * f(m, Heff_orig) */
    for(int i=0; i<N*N; i++){
        vec3 dm;

        llg_rhs(state, &m[i], &Heff_orig[i], &dm);
        m_pred[i] = add(m[i], scale(dm, dt));
        normalize(&m_pred[i]);
        
    }

    /* 3) compute Heff at predicted spins */
    effective_field(state, m_pred, Heff_pred);

    /* 4) corrector: m_new = m + 0.5*dt*( f(m,Heff_orig) + f(m_pred,Heff_pred) ) */
    for(int i=0; i<N*N; i++){
        vec3 dm_o, dm_p;

        llg_rhs(state, &m[i],      &Heff_orig[i], &dm_o);
        llg_rhs(state, &m_pred[i], &Heff_pred[i], &dm_p);

        vec3 dm_avg = scale(add(dm_o, dm_p), 0.5);

        m[i] = add(m[i], scale(dm_avg, dt));
        normalize(&m[i]);
    }
}

void init(SimulationState *state, const char *mode){
    /*
    Function to initialize the spin lattice with a specific configuration
    Available modes:
        "random"     -> fully random spins
        "vortex"     -> vortex-like pattern
        "quadrants"  -> four-domain configuration (+-z)
        "spiral"     -> spin spiral along x
        "bubble"     -> single reversed domain in the center
    */
    int N   = state->N;
    vec3 *m = state->m;

    srand((unsigned)time(NULL));

    // Center of the grid
    double cx = 0.5 * N;
    double cy = 0.5 * N;
    
    if (strcmp(mode, "random") == 0){
        for(int j=0; j<N; j++){
            for(int i=0; i<N; i++){
                
                int id = idx(i,j, N);
                
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
                int id = idx(i, j, N);
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
                int id = idx(i, j, N);

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
                int id = idx(i, j, N);
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

                int id = idx(i, j, N);
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
        exit(EXIT_FAILURE);
    }
}

double compute_energy(SimulationState *state)
{
    int N = state->N;
    vec3 *m = state->m;
    const Params *p = state->params;

    double E = 0.0;

    for(int j=0;j<N;j++){
        for(int i=0;i<N;i++){

            int id = idx(i,j,N);

            int ip = (i+1)%N;
            int jp = (j+1)%N;

            vec3 m0 = m[id];
            vec3 mx = m[idx(ip,j,N)];
            vec3 my = m[idx(i,jp,N)];

            /* exchange */
            vec3 dx = add(mx, scale(m0,-1));
            vec3 dy = add(my, scale(m0,-1));

            E += p->J * (dot(dx,dx) + dot(dy,dy));

            /* anisotropy */
            double proj = dot(m0, state->K_u_dir);
            E += -p->K_u * proj * proj;

            /* zeeman */
            vec3 H = {p->Hx,p->Hy,p->Hz};
            E += -dot(H,m0);
        }
    }

    return E;
}

/*===========================================================*/
/* Main                                                      */
/*===========================================================*/

int main(int argc, char *argv[]){

    
    Params params;
    default_params(&params);

    char mode[64]         = "random";
    char outdir[256]      = "run0";
    char energy_file[256] = "ene.csv";
    char path[512];

    int opt;

    while((opt = getopt(argc, argv, "N:t:s:J:D:K:d:x:y:z:X:Y:Z:m:o:e:h")) != -1){
        switch(opt){
            case 'N': params.N          = atoi(optarg); break;
            case 't': params.nsteps     = atoi(optarg); break;
            case 's': params.save_every = atoi(optarg); break;
            case 'J': params.J          = atof(optarg); break;
            case 'D': params.D          = atof(optarg); break;
            case 'K': params.K_u        = atof(optarg); break;
            case 'd': params.dt         = atof(optarg); break;
            case 'x': params.Hx         = atof(optarg); break;
            case 'y': params.Hy         = atof(optarg); break;
            case 'z': params.Hz         = atof(optarg); break;
            case 'X': params.kx         = atof(optarg); break;
            case 'Y': params.ky         = atof(optarg); break;
            case 'Z': params.kz         = atof(optarg); break;

            case 'm': strncpy(mode,optarg,63); break;
            case 'o': strncpy(outdir,optarg,255); break;
            case 'e': strncpy(energy_file,optarg,255); break;
            case 'h':
                printf("Usage: ./llg [options]\n");
                print_help(&params);
                exit(EXIT_FAILURE);

            default:
                printf("Usage: ./llg [options]\n");
                print_help(&params);
                exit(EXIT_FAILURE);
        }
    }

    mkdir(outdir, 0777);
    save_params(&params, mode, outdir);

    snprintf(path, sizeof(path), "%s/%s", outdir, energy_file);
    FILE *energy = fopen(path, "w");
    if(!energy){
        printf("Cannot open energy file\n");
        return 1;
    }

    SimulationState *state = sim_state_alloc(&params);
    if (!state) return 1;
    
    init(state, mode);

    for(int step=0; step<=params.nsteps; step++){
        llg_step_heun(state);

        if(step % params.save_every == 0){
            save_snapshot(state, step, outdir);
            // compute and save on file energy for monitoring
            double E = compute_energy(state);
            fprintf(energy, "%d,%.8f\n", step, E);

        }

        int barWidth    = 40;
        double progress = (double)step / params.nsteps;
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

    fclose(energy);
    state_free(state);

    return 0;
}