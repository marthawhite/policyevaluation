#ifndef _MATRIX_ALGORITHMS_H
#define _MATRIX_ALGORITHMS_H

#include "algorithm_utils.h"

// typedef the function pointers for updating to make it more readable
//typedef void  (*MatrixUpdateFcn)(gsl_vector *, gsl_vector *, const struct transition_info_t *, struct matrix_vars_t *);


// Algorithms that use A matrix could use different A matrices or share the same A
// For time reasons, we should share any A matrices instead of recomputing for each algorithm
// TODO: for now, for elegance, not implementing this time saving measure

struct matrix_alg_vars_t {
      
      // Primary weights
      gsl_vector * w;
      // Eligibility trace
      gsl_vector * e;

      // Function to update eligibility trace, e.g., accumulating or replacing
      TraceUpdateFcn update_trace;

      // Function to update key matrix
      MatrixUpdateFcn update_mat;
    
      // For some algorithms require additional working space, requires to init separately
      gsl_vector *work;
      gsl_vector *work1;   
 
      // For true A and C
      gsl_matrix *matA;
      gsl_matrix *matC;

      // For sketch maintain Ainv*z vector
      gsl_vector *Ainvz;

      // For algorithms that use an approximation to A
      struct matrix_vars_t * mvars;
    
      // For algorithms that use an approximation to C
      struct matrix_vars_t * mvarsC;
     
      int t;
      double F, D, I, M;
      double amat, bvec;
};

void compute_values_matrix(gsl_vector * values, const gsl_matrix * observations, const void * alg_vars, const char*name, const int sparse);

// Assumes name already in alg
void init_matrix_alg(struct alg_t * alg, const int numobservations);

void deallocate_matrix_alg(void * alg_vars);

void reset_matrix_alg(void * alg_vars);


int T_LSTD(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int LSTD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int PLSTD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int PrightLSTD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int PleftLSTD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int PleftAcc_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int PleftATD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int PPleftLSTD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int PB_Aw(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int B_Aw(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int B_PAPw(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int ATD_1storder(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int ATD_1storder_trueAC(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int ATD_2ndorder(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int EATD_2ndorder(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int ATD_2ndorder_tridiag(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int ATD_sketch(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int ATD_sketch_sm(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int ATD_sketch_sm1(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int ATD_sketch_qr(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int ATD_sketch_vec(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int ATD_2ndorder_trueA(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int ATD_2ndorder_fullA(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);

int LSTD_unit(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info);
#endif
