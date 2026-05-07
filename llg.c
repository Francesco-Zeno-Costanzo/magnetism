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
      + D ( m · (∇ x m) )
      − K_u (m · k)^2
      − H_ext · m
    ] dA


Terms:
1) Exchange interaction:
    E_ex = J |∇m|^2
2) Interfacial Dzyaloshinskii–Moriya interaction (DMI):
    E_DMI = D [ m · (∇ x m) ]
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
    double Hmax;          // Maximum field for hysteresis loop
    double Hmin;          // Minimum field for hysteresis loop
    double dH;            // Field increment for hysteresis loop
    double hx, hy, hz;    // Direction of the applied field for hysteresis loop
    double relax_tol;     // Tolerance for relaxation in hysteresis loop
    int max_steps;        // Maximum steps for relaxation in hysteresis loop
    int save_states;      // Flag to save states during hysteresis loop (1 to save, 0 to skip)
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
    p->N             = 64;
    p->nsteps        = 10000;
    p->save_every    = 50;
    p->J             = 1.0;
    p->gamma         = 1.76;
    p->alpha         = 0.1;
    p->K_u           = 1.0;
    p->D             = 1.0;
    p->dt            = 0.005;
    p->Hx            = 0.0;
    p->Hy            = 0.0;
    p->Hz            = 0.0;
    p->kx            = 0.0;
    p->ky            = 0.0;
    p->kz            = 0.0;
    p->Hmax          = 1.0;
    p->Hmin          = -1.0;
    p->dH            = 0.1;
    p->hx            = 0.0;
    p->hy            = 0.0;
    p->hz            = 0.0;
    p->relax_tol     = 1e-6;
    p->max_steps     = 40000;
    p->save_states   = 1;
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

void save_params(const Params *p, const char *sim, const char *mode, const char *outdir){
    /*
    Function to save the simulation parameters to a text file for record keeping
    Parameters:
        p: pointer to Params structure containing simulation parameters
        sim : const char *, simulation type (e.g. "single", "hysteresis")
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

    fprintf(f,"simulation_type %s\n", sim);

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

  
    fprintf(f,"\nHysteresis parameters\n");

    fprintf(f,"Hmax %.6f\n", p->Hmax);
    fprintf(f,"Hmin %.6f\n", p->Hmin);
    fprintf(f,"dH   %.6f\n", p->dH);

    fprintf(f,"\nField direction\n");

    fprintf(f,"hx_dir %.6f\n", p->hx);
    fprintf(f,"hy_dir %.6f\n", p->hy);
    fprintf(f,"hz_dir %.6f\n", p->hz);

    fprintf(f,"\nRelaxation\n");

    fprintf(f,"relax_tol   %.6e\n", p->relax_tol);
    fprintf(f,"relax_steps %d\n",   p->max_steps);

    fprintf(f,"\nSave states %d\n", p->save_states);

    fclose(f);
}

void print_help(Params *p){
    /* Help message */
    printf("Options:\n");

    printf(" -h  show this help message\n");
    printf(" -S  choice if single run or hysteresis loop (default single)\n");

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

    printf(" -m  initial condition (random [default], uniform, vortex...)\n");

    printf(" -o  output directory for saving snapshots.csv (default run0) \n");
    printf(" -e  energy file will be inside the output directory (default ene.csv) \n");

    printf("\nHysteresis loop parameters (only if -S hysteresis)\n");

    printf(" -H  maximum field (default %f)\n", p->Hmax);
    printf(" -a  minimum field (default %f)\n", p->Hmin);
    printf(" -b  field step dH (default %f)\n", p->dH);

    printf(" -i  field direction hx (default %f)\n", p->hx);
    printf(" -j  field direction hy (default %f)\n", p->hy);
    printf(" -k  field direction hz (default %f)\n", p->hz);

    printf(" -r  relaxation tolerance (default %e)\n", p->relax_tol);
    printf(" -R  relaxation max steps (default %d)\n", p->max_steps);

    printf(" -w  if 1 save final states at each field value (0/1)\n");

    printf("\nExamples:\n");

    printf("Simple simulation             : ./llg -N 128 -J 1 -D 1 -z 0.3 -m random -o run -e ene.csv \n\n");
    printf("Hysteresis loop hard direction: ./llg -S hysteresis -N 64 -H 1 -a -1 -b 0.1 -i 1 -X 0.1 -Y 0.9 -o run1\n");
    printf("Hysteresis loop easy direction: ./llg -S hysteresis -N 64 -H 1 -a -1 -b 0.1 -i 1 -X 0.9 -Y 0.1 -o run1\n");

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

            H_dmi.x = p->D * -(m_jp.z - m_jm.z)/2.0;
            H_dmi.y = p->D *  (m_ip.z - m_im.z)/2.0;
            H_dmi.z = p->D *  ((m_jp.x - m_jm.x) - (m_ip.y - m_im.y))/2.0;


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

double compute_energy(SimulationState *state){
    /*
    Function to compute the total energy of the system based on the current spin configuration

    Parameters:
        state: pointer to SimulationState structure containing current spins and parameters
    
    Returns:
        E: double representing the total energy of the system
    */
    int N = state->N;
    vec3 *m = state->m;
    const Params *p = state->params;

    double E = 0.0;

    for(int j=0;j<N;j++){
        for(int i=0;i<N;i++){

            int id = idx(i,j,N);

            int ip = (i+1)%N;
            int jp = (j+1)%N;
            int im = (i-1+N)%N;
            int jm = (j-1+N)%N;

            vec3 m0 = m[id];

            vec3 m_ip = m[idx(ip, j, N)];
            vec3 m_im = m[idx(im, j, N)];
            vec3 m_jp = m[idx(i, jp, N)];
            vec3 m_jm = m[idx(i, jm, N)];


            /* exchange */
            vec3 dx = add(m_ip, scale(m0,-1));
            vec3 dy = add(m_jp, scale(m0,-1));

            E += p->J * (dot(dx,dx) + dot(dy,dy));

            /* anisotropy */
            double proj = dot(m0, state->K_u_dir);
            E += -p->K_u * proj * proj;

            /* DMI */
            double dmy_dx = (m_ip.y - m_im.y) / 2.0;
            double dmx_dy = (m_jp.x - m_jm.x) / 2.0;
            double dmz_dx = (m_ip.z - m_im.z) / 2.0;
            double dmz_dy = (m_jp.z - m_jm.z) / 2.0;

            E += p->D * (m0.z * (dmy_dx - dmx_dy) + m0.x * dmz_dy - m0.y * dmz_dx);

            /* zeeman */
            vec3 H = {p->Hx,p->Hy,p->Hz};
            E += -dot(H,m0);
        }
    }

    return E;
}

void compute_mag(SimulationState *state, double *Mx, double *My, double *Mz){
    /*
    Function to compute the average magnetization components of the system
    Parameters:
        state: pointer to SimulationState structure containing current spins and parameters
        Mx, My, Mz: pointers to double where the average magnetization components will be stored
    */
    
    int N     = state->params->N;
    vec3 *m   = state->m;
    
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;

    for(int i = 0; i < N*N; i++){
        sx += m[i].x;
        sy += m[i].y;
        sz += m[i].z;
    }

    *Mx = sx/(N*N*1.0);
    *My = sy/(N*N*1.0);
    *Mz = sz/(N*N*1.0);

}

/*===========================================================*/
/* Functions to run simulations and store data               */
/*===========================================================*/

void run_simulation(const Params *p, const char *mode, const char *outdir, const char *energy_file){
    /*
    Function to run the LLG simulation with given parameters and initial mode
    Parameters:
        p: pointer to Params structure containing simulation parameters
        mode: const char *, initial configuration mode
        outdir: const char *, output directory path for saving results
        energy_file: const char *, name of the file to save energy values
    */

    char path[512];
    mkdir(outdir, 0777);
    save_params(p, "single", mode, outdir);

    snprintf(path, sizeof(path), "%s/%s", outdir, energy_file);
    FILE *energy = fopen(path, "w");
    if(!energy){
        printf("Cannot open energy file\n");
        exit(EXIT_FAILURE);
    }

    SimulationState *state = sim_state_alloc(p);
    if (!state) exit(EXIT_FAILURE);
    
    init(state, mode);

    for(int step=0; step<=p->nsteps; step++){
        llg_step_heun(state);

        if(step % p->save_every == 0){
            save_snapshot(state, step, outdir);
            // compute and save on file energy for monitoring
            double E = compute_energy(state);
            fprintf(energy, "%d,%.8f\n", step, E);

        }

        int barWidth    = 40;
        double progress = (double)step / p->nsteps;
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

}


void relax_system(SimulationState *state, double tol, int max_steps){
    /*
    Function to relax the system to a local energy minimum using the 
    LLG dynamics until the energy change is below a specified tolerance
    or a maximum number of steps is reached

    Parameters:
        state: pointer to SimulationState structure containing current spins and parameters
        tol: double, energy change tolerance for convergence
        max_steps: int, maximum number of steps to perform for relaxation
    */
    double E_old = compute_energy(state);
    double E_new = E_old;

    for(int step = 0; step < max_steps; step++){
        llg_step_heun(state);
        E_new = compute_energy(state);

        if(fabs(E_new - E_old) < tol){
            break;
        }

        E_old = E_new;
    }
}

void run_hysteresis(SimulationState *state, Params *p, const char *outdir){
    /*
    Function to run a hysteresis loop simulation by sweeping the external field
    from Hmax to Hmin and back, relaxing the system at each field step,
    and recording the magnetization and energy

    Parameters:
        state: pointer to SimulationState structure containing current spins and parameters
        p: pointer to Params structure containing simulation parameters (including field sweep parameters)
        outdir: const char *, output directory path for saving results
    */
    char fname[256];
    mkdir(outdir, 0777);
    sprintf(fname,"%s/hysteresis.csv",outdir);
    printf("Saving hysteresis loop to %s\n", fname);
    save_params(p, "hysteresis", "random", outdir);


    FILE *f = fopen(fname,"w");
    if(!f){
        printf("Cannot open hysteresis file\n");
        return;
    }

    double nx = p->hx;
    double ny = p->hy;
    double nz = p->hz;

    /* For plot simplicity */
    fprintf(f,"%d,%f,%f,%f,%d\n", 0, nx, ny, nz, 0);

    double n = sqrt(nx*nx + ny*ny + nz*nz);

    if(n == 0){
        printf("Invalid hysteresis direction\n");
        exit(1);
    }

    nx/=n;
    ny/=n;
    nz/=n;

    int step_id = 0;
    double Mx, My, Mz;

    /* Sweep up branch */
    for(double H = p->Hmax; H >= p->Hmin-p->dH; H -= p->dH){
        
        p->Hx = H * nx;
        p->Hy = H * ny;
        p->Hz = H * nz;

        printf("Field H = %.4f\n",H);

        relax_system(state, p->relax_tol, p->max_steps);

        compute_mag(state, &Mx, &My, &Mz);

        double E = compute_energy(state);

        fprintf(f,"%f,%f,%f,%f,%f\n",H,Mx,My,Mz,E);

        if(p->save_states){
            save_snapshot(state, step_id, outdir);
        }

        step_id++;
    }

    /* Sweep down branch */
    for(double H = p->Hmin; H <= p->Hmax+p->dH; H += p->dH){
        
        p->Hx = H * nx;
        p->Hy = H * ny;
        p->Hz = H * nz;

        printf("Field H = %.4f\n",H);

        relax_system(state, p->relax_tol, p->max_steps);

        compute_mag(state, &Mx, &My, &Mz);

        double E = compute_energy(state);

        fprintf(f,"%f,%f,%f,%f,%f\n",H,Mx,My,Mz,E);

        if(p->save_states){
            save_snapshot(state, step_id, outdir);
        }

        step_id++;
    }

    fclose(f);
}

/*===========================================================*/
/* Main                                                      */
/*===========================================================*/

int main(int argc, char *argv[]){

    Params params;
    default_params(&params);

    char sim[64]          = "single_run";
    char init_mode[64]    = "random";
    char outdir[256]      = "run0";
    char energy_file[256] = "ene.csv";
    

    int opt;

    while((opt = getopt(argc, argv, "S:N:t:s:J:D:K:d:x:y:z:X:Y:Z:m:o:e:H:a:b:i:j:k:r:R:w:h")) != -1){
        switch(opt){
            
            /*Type of simulation*/
            case 'S': strncpy(sim, optarg, 63); break;

            /* Simulation parameters */
            case 'N': params.N           = atoi(optarg); break;
            case 't': params.nsteps      = atoi(optarg); break;
            case 's': params.save_every  = atoi(optarg); break;
            case 'J': params.J           = atof(optarg); break;
            case 'D': params.D           = atof(optarg); break;
            case 'K': params.K_u         = atof(optarg); break;
            case 'd': params.dt          = atof(optarg); break;
            case 'x': params.Hx          = atof(optarg); break;
            case 'y': params.Hy          = atof(optarg); break;
            case 'z': params.Hz          = atof(optarg); break;
            case 'X': params.kx          = atof(optarg); break;
            case 'Y': params.ky          = atof(optarg); break;
            case 'Z': params.kz          = atof(optarg); break;
            case 'H': params.Hmax        = atof(optarg); break;
            case 'a': params.Hmin        = atof(optarg); break;
            case 'b': params.dH          = atof(optarg); break;
            case 'i': params.hx          = atof(optarg); break;
            case 'j': params.hy          = atof(optarg); break;
            case 'k': params.hz          = atof(optarg); break;
            case 'r': params.relax_tol   = atof(optarg); break;
            case 'R': params.max_steps   = atoi(optarg); break;
            case 'w': params.save_states = atoi(optarg); break;

            /* Simulation options */
            case 'm': strncpy(init_mode,  optarg, 63); break;
            case 'o': strncpy(outdir,     optarg,255); break;
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

    if (strcmp(sim, "single_run") == 0){
        run_simulation(&params, init_mode, outdir, energy_file);
    } 
    else if (strcmp(sim, "hysteresis") == 0){
        SimulationState *state = sim_state_alloc(&params);
        if (!state) exit(EXIT_FAILURE);

        init(state, init_mode);
        run_hysteresis(state, &params, outdir);

        state_free(state);
    }
    else{
        fprintf(stderr, "Unknown simulation type '%s'.\n", sim);
        exit(EXIT_FAILURE);
    }
    
    return 0;
}