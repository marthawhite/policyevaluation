#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "algorithm.h"
#include "matrix_algorithms.h"



// TODO: check parameters
const static struct{
      const char *name;
      AlgUpdateFcn update_fcn;
      MatrixUpdateFcn update_mat;
      TraceUpdateFcn update_trace;
} list_matrix_algorithms [] = {
      { "TLSTD", T_LSTD, update_mat_svd, update_trace_replacing},
      { "PLSTD", PLSTD_lambda, update_mat_sherman, update_trace_replacing},
      { "ATD2nd-TrueA", ATD_2ndorder_trueA, NULL, update_trace_replacing},
      { "ATD2nd-FullA", ATD_2ndorder_fullA, update_mat_sherman, update_trace_accumulating},
      { "ATD1st-TrueAC", ATD_1storder_trueAC, NULL, update_trace_accumulating},
      { "ATD2nd", ATD_2ndorder, update_mat_svd, update_trace_replacing},
      { "ATD1st", ATD_1storder, update_mat_svd, update_trace_accumulating},
      { "LSTD", LSTD_lambda, update_mat_sherman, update_trace_replacing},   
      { "ATD-TriDiag", ATD_2ndorder_tridiag, update_mat_tridiag, update_trace_replacing},
      { "ATD-Sketch-SM", ATD_sketch_sm, update_mat_sherman, update_trace_replacing},
      { "ATD-Sketch-SM1", ATD_sketch_sm1, update_mat_normal, update_trace_replacing},
      { "ATD-Sketch", ATD_sketch, update_mat_sherman, update_trace_accumulating},
      { "ATD-Sketch-Vec", ATD_sketch_vec, update_mat_sherman, update_trace_accumulating},
      { "Pone-Acc", PleftAcc_lambda, update_mat_normal, update_trace_replacing},
      { "Pone-ATD", PleftATD_lambda, update_mat_normal, update_trace_replacing},
      { "Pone-LSTD", PleftLSTD_lambda, update_mat_normal, update_trace_replacing},
      { "PPone-LSTD", PPleftLSTD_lambda, update_mat_normal, update_trace_replacing},
      { "PB-Aw", PB_Aw, update_mat_normal, update_trace_replacing},
      { "B-Aw", B_Aw, update_mat_normal, update_trace_replacing},
      { "B-PAPw", B_PAPw, update_mat_normal, update_trace_replacing},
      { "EATD2nd", EATD_2ndorder, update_mat_svd, update_trace_accumulating},
      { "LSTD-Unit", LSTD_unit, NULL, NULL},
};
const static int num_totalalgs = (sizeof(list_matrix_algorithms) / sizeof(list_matrix_algorithms[0]));


void init_matrix_alg(struct alg_t * alg, const int numobservations){

      // alg_vars is a void *, initialized here to the correct struct size
      alg->alg_vars = calloc(1, sizeof(struct matrix_alg_vars_t));
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg->alg_vars;

      // TODO: different way to give numfeatures, right now assuming MDP giving observations = features
      int numfeatures = numobservations;
   
      vars->w = gsl_vector_calloc(numfeatures);
      vars->e = gsl_vector_calloc(numfeatures);

      int i;
      // Get the update functions for this algorithm
      for (i = 0; i < num_totalalgs; i++) {
         //printf("name is %s\n", alg->name);
         if (!strcmp(list_matrix_algorithms[i].name, alg->name)) {
            alg->update_fcn = list_matrix_algorithms[i].update_fcn;
            alg->reset_fcn = reset_matrix_alg;
            vars->update_trace = list_matrix_algorithms[i].update_trace;
            vars->update_mat = list_matrix_algorithms[i].update_mat;
            break;
         }
      }
      // All algorithm use a dot product to compute the value function
      alg->get_values = compute_values_matrix;
      
      vars->mvars = NULL;
      vars->mvarsC = NULL;
      vars->work = NULL;
      vars->Ainvz = NULL;
      vars->matA = NULL;
      vars->matC = NULL;

      vars->F = 0;
      vars->D = 0;
      vars->I = 1;
      vars->M = 0;
      vars->amat = 0;
      vars->bvec = 0;
      // TODO: find a better way to set mvar_params
      struct mvar_params_t mvar_params;
      mvar_params.r = 50;
      mvar_params.max_r = 2*mvar_params.r;
      mvar_params.threshold = 0.01;
           
      /*
       * Algorithm specific initializations
       */
      // TODO: currently, ATD2nd would be matched for anything starting with ATD2nd
      // should change to use strncmp
      if (strcmp(alg->name, "TLSTD") == 0) {
         const char * mattype = "low_rank";
         vars->mvars = allocate_matrix_vars(mattype, numfeatures, &mvar_params);
      }
      else if(strcmp(alg->name, "ATD2nd") == 0 || strcmp(alg->name, "EATD2nd") == 0 ){
         const char * mattype = "low_rank";
         vars->mvars = allocate_matrix_vars(mattype, numfeatures, &mvar_params);
      }
      else if(strcmp(alg->name, "ATD1st") == 0){
          const char * mattype = "low_rank";
          vars->work = gsl_vector_alloc(numfeatures);
          vars->mvars = allocate_matrix_vars(mattype, numfeatures, &mvar_params);
          vars->mvarsC = allocate_matrix_vars(mattype, numfeatures, &mvar_params);
      }
      else if(strcmp(alg->name, "ATD2nd-TrueA") == 0){
          vars->matA = gsl_matrix_calloc(numfeatures, numfeatures);
      }
      else if(strcmp(alg->name, "ATD1st-TrueAC") == 0){
          vars->work = gsl_vector_alloc(numfeatures);
          vars->matA = gsl_matrix_calloc(numfeatures, numfeatures);
          vars->matC = gsl_matrix_calloc(numfeatures, numfeatures);
      }
      else if(strcmp(alg->name, "LSTD") == 0 || strcmp(alg->name, "ATD2nd-FullA") == 0){
          const char * mattype = "full";
          vars->mvars = allocate_matrix_vars(mattype, numfeatures, &mvar_params);
      }
      else if(strcmp(alg->name, "ATD-TriDiag") == 0){
          const char * mattype = "tridiag";
          vars->mvars = allocate_matrix_vars(mattype, numfeatures, &mvar_params);
      }
      else if(strcmp(alg->name, "B-PAPw") == 0 || strcmp(alg->name, "ATD-Sketch-SM") == 0 || strcmp(alg->name, "ATD-Sketch-Vec") == 0 || strcmp(alg->name, "PLSTD") == 0){
          const char * mattype = "atdsketch";
          //mvar_params.r = 100;
          //mvar_params.max_r = 2*mvar_params.r;
          //mvar_params.threshold = 0.01;
          vars->Ainvz = gsl_vector_calloc(numfeatures);
          vars->work = gsl_vector_alloc(numfeatures);
          vars->mvars = allocate_matrix_vars(mattype, numfeatures, &mvar_params);
      }
      else if(strcmp(alg->name, "ATD-Sketch") == 0){
          const char * mattype = "atdsketch";
          vars->mvars = allocate_matrix_vars(mattype, numfeatures, &mvar_params);
      }
      else if(strcmp(alg->name, "Pone-LSTD") == 0 || strcmp(alg->name, "Pone-Acc") == 0  || strcmp(alg->name, "Pone-ATD") == 0){
          const char * mattype = "onesidesketch";
          vars->work = gsl_vector_alloc(numfeatures);
          vars->work1 = gsl_vector_alloc(mvar_params.r);
          vars->mvars = allocate_matrix_vars(mattype, numfeatures, &mvar_params);
      }
      else if(strcmp(alg->name, "PPone-LSTD") == 0 || strcmp(alg->name, "B-Aw") == 0){
          const char * mattype = "full";
          vars->work = gsl_vector_alloc(numfeatures);
          vars->mvars = allocate_matrix_vars(mattype, numfeatures, &mvar_params);
          vars->work1 = gsl_vector_alloc(mvar_params.r);        
      }
      //printf("algorithm found\n");
}

void deallocate_matrix_alg(void * alg_vars){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;
    
      gsl_vector_free(vars->w);
      gsl_vector_free(vars->e);
      
      if (vars->mvars != NULL) {
         deallocate_matrix_vars(vars->mvars);
      }
      if (vars->mvarsC != NULL) {
         deallocate_matrix_vars(vars->mvarsC);
      }
      if (vars->work != NULL) {
         gsl_vector_free(vars->work);
      }
      if (vars->work1 != NULL) {
         gsl_vector_free(vars->work1);
      }
      if (vars->matA != NULL) {
         gsl_matrix_free(vars->matA);
      }
      if (vars->matC != NULL) {
         gsl_matrix_free(vars->matC);
      }
      if (vars->Ainvz != NULL) {
         gsl_vector_free(vars->Ainvz);
      }
      /*if (alg_vars->projmat != NULL){
         gsl_matrix_free(alg_vars->projmat);
         gsl_vector_free(alg_vars->px_t);
         gsl_vector_free(alg_vars->px_tp1);
      }
      if (alg_vars->batchxt != NULL){
         gsl_matrix_free(alg_vars->batchxt);
         gsl_matrix_free(alg_vars->batchxtp1);
         gsl_vector_free(alg_vars->batchrewards);
      }*/    
}

void reset_matrix_alg(void * alg_vars){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;

      gsl_vector_set_zero(vars->w);
      gsl_vector_set_zero(vars->e);

      vars->t = 0;

      vars->I = 1;
      vars->M = 0;
      vars->F = 0;
      vars->D = 0;
      
      if (vars->mvars != NULL) {
         reset_matrix_vars(vars->mvars);
      }
      if (vars->mvarsC != NULL) {
         reset_matrix_vars(vars->mvarsC);
      }
      if (vars->work != NULL) {
         gsl_vector_set_zero(vars->work);
      }
      if (vars->work1 != NULL) {
         gsl_vector_set_zero(vars->work1);
      }
      if (vars->Ainvz != NULL){
         gsl_vector_set_zero(vars->Ainvz);
      }
      /*if (vars->batchxt != NULL){
         gsl_matrix_set_zero(alg_vars->batchxt);
         gsl_matrix_set_zero(alg_vars->batchxtp1);
         gsl_vector_set_zero(alg_vars->batchrewards);
      }
      if(alg_vars->projmat != NULL){
          struct rgen_t rt;
          const gsl_rng_type * T;
          T = gsl_rng_default;
          rt.r = gsl_rng_alloc (T);
          generate_random_matrix(alg_vars->projmat, 1.0/sqrt((double)alg_vars->px_t->size), 0.0, &rt);
          gsl_rng_free(rt.r);
          gsl_vector_set_zero(alg_vars->px_t);
          gsl_vector_set_zero(alg_vars->px_tp1);
      }
      */
}


// Assumes that the SVD of A and b was updated outside of this function
int T_LSTD(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info) {
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;

      //update z vector
      vars->update_trace(vars->e, params, info);
    
      update_bvec(vars->e, info, vars->mvars);
    
      compute_dvec(vars->mvars->dvec, info);
    
      vars->update_mat(vars->e, vars->mvars->dvec, info, vars->mvars);
      // compute weight vector
      compute_weights(vars->w, vars->mvars);
  
      return 0;
}

int ATD_2ndorder_trueA(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;

    vars->update_trace(vars->e, params, info);
    
    double delta = compute_delta(vars->w, info);
    
    gsl_blas_dgemv (CblasNoTrans, delta*params->alpha_t/((double)vars->mvars->t + 1.0), vars->matA, vars->e, 1.0, vars->w);
    
    gsl_blas_daxpy(delta*params->beta_t, vars->e, vars->w);
    
    vars->mvars->t++;
    
    return 0;
}

int ATD_2ndorder_fullA(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;

    if (vars->mvars->t == 0) {
        gsl_vector_view diagA = gsl_matrix_diagonal(vars->mvars->mat_main);
        gsl_vector_set_all(&diagA.vector, params->eta_t);
    }

    vars->update_trace(vars->e, params, info);

    double delta = compute_delta(vars->w, info);

    compute_dvec(vars->mvars->dvec, info);

    vars->update_mat(vars->e, vars->mvars->dvec, info, vars->mvars);

    double stepsize = 1.0/((double)vars->mvars->t);    

    update_weights(vars->w, vars->e, delta*stepsize, 1.0, vars->mvars, MAT_FULL);   

    gsl_blas_daxpy(delta*0.00001, vars->e, vars->w);

    vars->mvars->t++;

    return 0;
}

int ATD_2ndorder(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;

    if (vars->mvars->t == 0) {
        vars->mvars->threshold = 0.01;
    }   
    vars->update_trace(vars->e, params, info);
    double delta = compute_delta(vars->w, info);
   
    compute_dvec(vars->mvars->dvec, info);
  
    //if(vars->mvars->t % 50) printf("the condition number is %f\n", gsl_vector_get(vars->mvars->sigmavec, 0)*params->threshold); 
    //printf("the current rank is %d\n", get_rank(vars->mvars->sigmavec, params->threshold));
    
    vars->update_mat(vars->e, vars->mvars->dvec, info, vars->mvars);

    double stepsize = 1.0/((double)vars->mvars->t);

    update_weights(vars->w, vars->e, delta*stepsize, 1.0, vars->mvars, MAT_SVD_INV);
    
    gsl_blas_daxpy(delta*params->beta_t, vars->e, vars->w);
    
    return 0;
}

int EATD_2ndorder(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){

    struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;

    //NOTE: this rho should be rho_{t-1}, does not matter for on policy
    vars->F = info->rho_t*info->gamma_t*vars->F + vars->I;

    vars->M = params->lambda_t*vars->I + (1 - params->lambda_t)*vars->F;

    gsl_vector_scale(vars->e, info->gamma_t*info->rho_t*params->lambda_t);

    gsl_blas_daxpy(vars->M*info->rho_t, info->x_t, vars->e);

    double delta = compute_delta(vars->w, info);

    compute_dvec(vars->mvars->dvec, info);

    vars->update_mat(vars->e, vars->mvars->dvec, info, vars->mvars);

    double stepsize = 1.0/((double)vars->mvars->t);

    update_weights(vars->w, vars->e, delta*stepsize, 1.0, vars->mvars, MAT_SVD_INV);

    gsl_blas_daxpy(delta*params->beta_t, vars->e, vars->w);

    return 1;
}

int ATD_2ndorder_tridiag(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
 
    struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;
    
    if (vars->mvars->t == 0) {
        gsl_vector_set_all(vars->mvars->diagA, params->eta_t);
    }

    vars->update_trace(vars->e, params, info);  
 
    compute_dvec(vars->mvars->dvec, info);
     
    double delta = compute_delta(vars->w, info);

    vars->update_mat(vars->e, vars->mvars->dvec, info, vars->mvars);

    double stepsize = 1.0/((double)vars->mvars->t);

    update_weights(vars->w, vars->e, delta*stepsize, 1.0, vars->mvars, MAT_TRI_DIAG_INV);

    return 1;
}

int ATD_sketch(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
    struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;
    if (vars->mvars->t == 0) {
      gsl_vector_view diagA = gsl_matrix_diagonal(vars->mvars->mat_main);
      gsl_vector_set_all(&diagA.vector, params->eta_t); 
      int batchsize = vars->mvars->projmat->size1/vars->mvars->projmat->size2;
      generate_aggregate_matrix(vars->mvars->projmat, batchsize);      

      /*gsl_matrix *gsmat = gsl_matrix_alloc(vars->e->size, 10000);
      generate_random_matrix(gsmat, 1.0/sqrt(gsmat->size2), 0, vars->mvars->rt);
      gsl_matrix *workagg = gsl_matrix_alloc(10000, vars->mvars->projmat->size2);
      int batchsize = workagg->size1/vars->mvars->projmat->size2;
      generate_aggregate_matrix(workagg, batchsize);
 
      gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, gsmat, workagg, 0.0, vars->mvars->projmat);
     
      gsl_matrix_free(gsmat); 
      gsl_matrix_free(workagg);
      */
    }
    vars->update_trace(vars->e, params, info);
   
    double delta = compute_delta(vars->w, info);

    compute_dvec(vars->mvars->dvec, info);

    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->e, 0.0, vars->mvars->work2);
     
    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->mvars->dvec, 0.0, vars->mvars->work3);
    
    vars->update_mat(vars->mvars->work2, vars->mvars->work3, info, vars->mvars);
    double stepsize = 1.0/((double)vars->mvars->t);

    update_weights(vars->w, vars->e, delta*stepsize, 1.0, vars->mvars, MAT_SKET_FULL); 
 
    gsl_blas_daxpy(delta*0.00001, vars->e, vars->w);

    return 0;
}

int ATD_sketch_sm(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){

    struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;
   
    if (vars->mvars->t == 0) {
      gsl_vector_view diagA = gsl_matrix_diagonal(vars->mvars->mat_main);
      gsl_vector_set_all(&diagA.vector, params->eta_t);
      //0.056 energy 
      //gsl_matrix *work_mat = gsl_matrix_alloc(vars->e->size, vars->e->size);
      //generate_srht_matrix(vars->mvars->projmat, work_mat, vars->mvars->rt);
      //gsl_matrix_free(work_mat);
      generate_random_matrix(vars->mvars->projmat, 1.0/sqrt(vars->mvars->projmat->size2), 0, vars->mvars->rt);
    }
    vars->update_trace(vars->e, params, info);
    //gsl_vector_print(info->x_t);
    double delta = compute_delta(vars->w, info);
    compute_dvec(vars->mvars->dvec, info);
    
    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->e, 0.0, vars->mvars->work2); 

    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->mvars->dvec, 0.0, vars->mvars->work3);

    vars->update_mat(vars->mvars->work2, vars->mvars->work3, info, vars->mvars);

    double stepsize = 1.0/(double)vars->mvars->t;

    update_weights(vars->w, vars->e, delta*stepsize, 1.0, vars->mvars, MAT_SKET_FULL);

    gsl_blas_daxpy(delta*0.00001, vars->e, vars->w);

    return 0;
}

int ATD_sketch_sm1(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){

    struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;

    if (vars->mvars->t == 0) {
      gsl_vector_view diagA = gsl_matrix_diagonal(vars->mvars->mat_cov);
      gsl_vector_set_all(&diagA.vector, params->eta_t);
      //generate_random_matrix(vars->mvars->projmat, 1.0/sqrt(vars->mvars->projmat->size2), 0, vars->mvars->rt);
      generate_countsketch_matrix(vars->mvars->projmat, vars->mvars->rt);
      //gsl_matrix *work_mat = gsl_matrix_alloc(vars->w->size, vars->w->size);
      //generate_srht_matrix(vars->mvars->projmat, work_mat, vars->mvars->rt);
      //gsl_matrix_free(work_mat);
    }

    vars->update_trace(vars->e, params, info);

    double delta = compute_delta(vars->w, info);

    compute_dvec(vars->mvars->dvec, info);

    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->e, 0.0, vars->mvars->work2);

    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->mvars->dvec, 0.0, vars->mvars->work3);

    update_mat_cov(vars->mvars->work2, vars->mvars->work3, info, vars->mvars);

    vars->update_mat(vars->mvars->work2, vars->mvars->work3, info, vars->mvars);

    double stepsize = 1.0/(double)vars->mvars->t;
    update_weights(vars->w, vars->e, delta*stepsize, 1.0, vars->mvars, MAT_SKET_QR_INV);
    gsl_blas_daxpy(delta*params->beta_t, vars->e, vars->w);

    return 0;
}

int ATD_sketch_qr(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){

    struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;

    if (vars->mvars->t == 0) {
        generate_random_matrix(vars->mvars->projmat, 1.0/sqrt(vars->mvars->projmat->size2), 0, vars->mvars->rt); 
        gsl_matrix_memcpy(vars->mvars->projmat1, vars->mvars->projmat);
    }   

    vars->update_trace(vars->e, params, info);

    double delta = compute_delta(vars->w, info);

    compute_dvec(vars->mvars->dvec, info);
  
    update_mat_svd(vars->e, vars->mvars->dvec, info, vars->mvars);   
 
    if(vars->mvars->matu->size2 == vars->mvars->projmat->size2)
    {
       gsl_matrix_memcpy(vars->mvars->projmat, vars->mvars->matu);
       gsl_matrix_memcpy(vars->mvars->projmat1, vars->mvars->matv);
    }
    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat1, vars->e, 0.0, vars->mvars->work2);

    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->mvars->dvec, 0.0, vars->mvars->work3);

    vars->update_mat(vars->mvars->work2, vars->mvars->work3, info, vars->mvars);

    double stepsize = 1.0/(double)vars->mvars->t;

    update_weights(vars->w, vars->e, delta*stepsize, 1.0, vars->mvars, MAT_SKET_FULL_TWO);

    gsl_blas_daxpy(delta*params->alpha_t/100, vars->e, vars->w);

    return 0;
}

int ATD_sketch_vec(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
    struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;

    if (vars->mvars->t == 0) {
      gsl_vector_view diagA = gsl_matrix_diagonal(vars->mvars->mat_main);
      gsl_vector_set_all(&diagA.vector, params->eta_t); 
      generate_random_matrix(vars->mvars->projmat, 1.0/sqrt(vars->mvars->projmat->size2), 0, vars->mvars->rt);     
    }
    vars->update_trace(vars->e, params, info);
    double delta = compute_delta(vars->w, info);
    
    compute_dvec(vars->mvars->dvec, info);

    update_Ainvz(vars->Ainvz, info, params, vars->mvars);

    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->e, 0.0, vars->mvars->work2);

    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->mvars->dvec, 0.0, vars->mvars->work3);

    vars->update_mat(vars->mvars->work2, vars->mvars->work3, info, vars->mvars);

    double stepsize = 1.0/(double)vars->mvars->t;

    gsl_blas_daxpy(delta*stepsize, vars->Ainvz, vars->w);

    gsl_blas_daxpy(delta*0.00001, vars->e, vars->w);

    return 0;
}


int ATD_1storder_trueAC(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;

    //update z vector
    vars->update_trace(vars->e, params, info);
    
    //compute delta
    double delta = compute_delta(vars->w, info);
    
    gsl_blas_dgemv (CblasNoTrans, 1.0, vars->matC, vars->e, 0.0, vars->work);
    
    gsl_blas_dgemv (CblasTrans, delta*params->alpha_t/((double)vars->mvars->t + 1.0), vars->matA, vars->work, 1.0, vars->w);
    
    vars->mvars->t++;
    //printf("the weight vector in 1st is:\n");
    //gsl_vector_print(vars->w);
    return 0;
}

int ATD_1storder(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;

      //update z vector
    vars->update_trace(vars->e, params, info);
    
    double delta = compute_delta(vars->w, info);
    
    compute_dvec(vars->mvars->dvec, info);
    
    vars->update_mat(vars->e, vars->mvars->dvec, info, vars->mvars);
    
    vars->update_mat(info->x_t, info->x_t, info, vars->mvarsC);
    
    //NOTE: working vectors in mvars cannot emerge in this file to avoid conflicts
    op_mat_vector_mul(vars->work, vars->e, vars->mvarsC, MAT_SVD_INV);
    
    update_weights(vars->w, vars->work, delta*params->alpha_t/((double)vars->mvars->t), 1.0, vars->mvars, MAT_SVD_TRANS);
    
    return 1;
}

// modify: can do both svd or directly maintain A^inv
int LSTD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;
    
    if (vars->mvars->t == 0) {
      gsl_vector_view diagA = gsl_matrix_diagonal(vars->mvars->mat_main);
      gsl_vector_set_all(&diagA.vector, params->eta_t);
      //vars->mvars->threshold = params->threshold;
      //generate_countsketch_matrix(vars->mvars->projmat, vars->mvars->rt);
      //generate_random_matrix(vars->mvars->projmat, 1.0/sqrt(vars->mvars->projmat->size2), 0, vars->mvars->rt);
    }
    vars->update_trace(vars->e, params, info);
    
    update_bvec(vars->e, info, vars->mvars);
    
    compute_dvec(vars->mvars->dvec, info);

    vars->update_mat(vars->e, vars->mvars->dvec, info, vars->mvars);
    
    //if(vars->mvars->t % 100 ==0)
      update_weights(vars->w, vars->mvars->bvec, 1, 0, vars->mvars, MAT_FULL);
       //op_mat_vector_mul(vars->w, vars->mvars->bvec, vars->mvars, MAT_FULL_INV);

    return 1;
}

int PLSTD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;

    if (vars->mvars->t == 0) {
        gsl_vector_view diagA = gsl_matrix_diagonal(vars->mvars->mat_main);
        gsl_vector_set_all(&diagA.vector, params->eta_t);
        generate_random_matrix(vars->mvars->projmat, 1.0/sqrt(vars->mvars->projmat->size2), 0, vars->mvars->rt);
    }
    vars->update_trace(vars->e, params, info);

    update_bvec(vars->e, info, vars->mvars);

    compute_dvec(vars->mvars->dvec, info);

    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->e, 0.0, vars->mvars->work2);

    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->mvars->dvec, 0.0, vars->mvars->work3);   
    vars->update_mat(vars->mvars->work2, vars->mvars->work3, info, vars->mvars);

    op_mat_vector_mul(vars->w, vars->mvars->bvec, vars->mvars, MAT_SKET_FULL);

    return 1;
}

int PleftLSTD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;
    if (vars->mvars->t == 0) {
        vars->mvars->threshold = 0.000001;
        generate_random_matrix(vars->mvars->projmat, 1.0/sqrt(vars->mvars->projmat->size2), 0, vars->mvars->rt);
    }
    //printf("entered p one lstd \n");
    vars->update_trace(vars->e, params, info);

    update_bvec(vars->e, info, vars->mvars);

    compute_dvec(vars->mvars->dvec, info);
    //left side projection, maintain transpose of A
    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->e, 0.0, vars->mvars->work);
    vars->update_mat(vars->mvars->dvec, vars->mvars->work, info, vars->mvars);
    
    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->mvars->bvec, 0.0, vars->mvars->work2);

    if(vars->mvars->t % 100 == 0)
    {  
       op_mat_vector_mul(vars->w, vars->mvars->work2, vars->mvars, MAT_FULL_TRAN_INV);
    }
    return 1;
}


int PleftAcc_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;
    if (vars->mvars->t == 0) {
        generate_random_matrix(vars->mvars->projmat, 1.0/sqrt(vars->mvars->projmat->size2), 0, vars->mvars->rt);
        gsl_vector_view diagA = gsl_matrix_diagonal(vars->mvars->mat_cov);
        gsl_vector_set_all(&diagA.vector, 1);
        vars->mvars->t = 1;
        vars->t = 0;
    }
    //printf("entered p one lstd \n");
    vars->update_trace(vars->e, params, info);
    update_bvec(vars->e, info, vars->mvars);
    compute_dvec(vars->mvars->dvec, info);
    
    //this is for eliminating numerical issue, later should scale back when update main matrix
    double scalor_z = 1;
    scalor_z = 1.0/(double)vars->mvars->t;
    gsl_blas_dgemv(CblasTrans, 1.0, vars->mvars->projmat, vars->e, 0.0, vars->mvars->work);
    gsl_vector_scale(vars->mvars->work, scalor_z);       

    double dd = pow(gsl_blas_dnrm2(vars->mvars->dvec), 2);
    //then update the covariance matrix (SA)(SA)^T
    //first compute Ad
    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->mat_main, vars->mvars->dvec, 0.0, vars->work1);
    double alpha = vars->mvars->t-1>0? pow((double)(vars->mvars->t-1)/(double)vars->mvars->t,2):1;
    double beta = dd;
    update_sherman_general(vars->mvars->mat_cov, vars->mvars->work, vars->mvars->work, alpha, beta, vars->mvars);

    alpha = 1, beta = (double)(vars->mvars->t-1)/(double)vars->mvars->t;
    if(vars->mvars->t > 1){//otherwise the main mat is zero
      update_sherman_general(vars->mvars->mat_cov, vars->work1, vars->mvars->work, alpha, beta, vars->mvars);
      update_sherman_general(vars->mvars->mat_cov, vars->mvars->work, vars->work1, alpha, beta, vars->mvars); 
    }
    //then update the main matrix, maintain the transpose 
    gsl_vector_scale(vars->mvars->work, 1.0/scalor_z);
    vars->update_mat(vars->mvars->dvec, vars->mvars->work, info, vars->mvars);
    //finally compute solution
    gsl_blas_dgemv(CblasTrans, 1.0, vars->mvars->projmat, vars->mvars->bvec, 0.0, vars->mvars->work);
    gsl_blas_dgemv(CblasNoTrans, 1.0, vars->mvars->mat_cov, vars->mvars->work, 0.0, vars->work1); 
    gsl_blas_dgemv(CblasNoTrans, 1.0, vars->mvars->mat_main, vars->work1, 0.0, vars->w);

    return 1;
}

int PleftATD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;
    if (vars->mvars->t == 0) {
        generate_random_matrix(vars->mvars->projmat, 1.0/sqrt(vars->mvars->projmat->size2), 0, vars->mvars->rt);
        gsl_vector_view diagA = gsl_matrix_diagonal(vars->mvars->mat_cov);
        gsl_vector_set_all(&diagA.vector, 1);
        vars->mvars->t = 1;
        vars->t = 0;
    }
    //printf("entered p one lstd \n");
    vars->update_trace(vars->e, params, info);
    compute_dvec(vars->mvars->dvec, info);
    double delta = compute_delta(vars->w, info);

    //this is for eliminating numerical issue, later should scale back when update main matrix
    double scalor_z = 1;
    scalor_z = 1.0/(double)vars->mvars->t;
    gsl_blas_dgemv(CblasTrans, 1.0, vars->mvars->projmat, vars->e, 0.0, vars->mvars->work);
    gsl_vector_scale(vars->mvars->work, scalor_z);       

    double dd = pow(gsl_blas_dnrm2(vars->mvars->dvec), 2);
    //then update the covariance matrix (SA)(SA)^T
    //first compute Ad
    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->mat_main, vars->mvars->dvec, 0.0, vars->work1);
    double alpha = vars->mvars->t-1>0? pow((double)(vars->mvars->t-1)/(double)vars->mvars->t,2):1;
    double beta = dd;
    update_sherman_general(vars->mvars->mat_cov, vars->mvars->work, vars->mvars->work, alpha, beta, vars->mvars);

    alpha = 1, beta = (double)(vars->mvars->t-1)/(double)vars->mvars->t;
    if(vars->mvars->t > 1){//otherwise the main mat is zero
      update_sherman_general(vars->mvars->mat_cov, vars->work1, vars->mvars->work, alpha, beta, vars->mvars);
      update_sherman_general(vars->mvars->mat_cov, vars->mvars->work, vars->work1, alpha, beta, vars->mvars); 
    }
    //then update the main matrix, maintain the transpose 
    gsl_vector_scale(vars->mvars->work, 1.0/scalor_z);
    vars->update_mat(vars->mvars->dvec, vars->mvars->work, info, vars->mvars);
    //finally compute solution
    gsl_blas_dgemv(CblasNoTrans, 1.0, vars->mvars->mat_cov, vars->mvars->work, 0.0, vars->work1); 
    gsl_blas_dgemv(CblasNoTrans, 1.0/(double)(vars->mvars->t-1)*delta, vars->mvars->mat_main, vars->work1, 1.0, vars->w);
    //gsl_blas_dgemv(CblasNoTrans, params->alpha_t*delta, vars->mvars->mat_main, vars->work1, 1.0, vars->w);

    //use regularizer
    gsl_blas_daxpy(delta*params->beta_t, vars->e, vars->w);

    return 1;
}



int PPleftLSTD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;
    if (vars->mvars->t == 0) {
        vars->mvars->threshold = params->threshold;
        generate_random_matrix(vars->mvars->projmat, 1.0/sqrt(vars->mvars->projmat->size2), 0, vars->mvars->rt);
    }
    vars->update_trace(vars->e, params, info);

    update_bvec(vars->e, info, vars->mvars);

    compute_dvec(vars->mvars->dvec, info);

    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->e, 0.0, vars->work1);
    
    gsl_blas_dgemv (CblasNoTrans, 1.0, vars->mvars->projmat, vars->work1, 0.0, vars->work);

    vars->update_mat(vars->work, vars->mvars->dvec, info, vars->mvars);

    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->mvars->bvec, 0.0, vars->work1);

    gsl_blas_dgemv (CblasNoTrans, 1.0, vars->mvars->projmat, vars->work1, 0.0, vars->work);

    if(vars->mvars->t % 2000 == 0)
        op_mat_vector_mul(vars->w, vars->work, vars->mvars, MAT_FULL_INV);

    return 1;
}


int PrightLSTD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;
    if (vars->mvars->t == 0) {
        vars->mvars->threshold = params->threshold;
        generate_random_matrix(vars->mvars->projmat, 1.0/sqrt(vars->mvars->projmat->size2), 0, vars->mvars->rt);
        gsl_matrix_memcpy(vars->mvars->matu, vars->mvars->projmat);
        gsl_linalg_SV_decomp (vars->mvars->matu, vars->mvars->matv, vars->mvars->sigmavec, vars->mvars->work);
    }
    vars->update_trace(vars->e, params, info);

    update_bvec(vars->e, info, vars->mvars);

    compute_dvec(vars->mvars->dvec, info);

    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->mvars->dvec, 0.0, vars->mvars->work);
    
    vars->update_mat(vars->e, vars->mvars->work, info, vars->mvars);
    
    if(vars->mvars->t % 100 == 0){
       op_mat_vector_mul(vars->mvars->work, vars->mvars->bvec, vars->mvars, MAT_FULL_INV);
       compute_diagonal_inverse(vars->mvars->work1, vars->mvars->sigmavec, 0);
       gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->matv, vars->mvars->work, 0.0, vars->mvars->work2);  
       gsl_vector_mul(vars->mvars->work1, vars->mvars->work2);
       gsl_blas_dgemv (CblasNoTrans, 1.0, vars->mvars->matu, vars->mvars->work1, 0.0, vars->w); 
       //printf("solution computed\n");
       //gsl_vector_print(vars->mvars->sigmavec);
    }
    return 1;
}

int PB_Aw(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;
    if (vars->mvars->t == 0) { 
       generate_random_matrix(vars->mvars->projmat, 1.0/sqrt(vars->mvars->projmat->size2), 0, vars->mvars->rt);
      //gsl_matrix *work_mat = gsl_matrix_alloc(vars->e->size, vars->e->size);
      //generate_srht_matrix(vars->mvars->projmat, work_mat, vars->mvars->rt);
      //gsl_matrix_free(work_mat);
       /*gsl_matrix_memcpy(vars->mvars->delta_main, vars->mvars->projmat);
       gsl_linalg_SV_decomp (vars->mvars->delta_main, vars->mvars->work_mat_main, vars->mvars->work2, vars->mvars->work);
       for(int i = 0; i<vars->mvars->r; i++){
          if(gsl_vector_get(vars->mvars->work2, i) > 0.00001)
                  gsl_vector_set(vars->mvars->work2, i, 1.0/gsl_vector_get(vars->mvars->work2, i));
          else gsl_vector_set(vars->mvars->work2, i, 0);
       }*/
    }
    vars->update_trace(vars->e, params, info);

    update_bvec(vars->e, info, vars->mvars);

    compute_dvec(vars->mvars->dvec, info);

    double delta = compute_delta(vars->w, info);

    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->e, 0.0, vars->mvars->work);

    vars->update_mat(vars->mvars->dvec, vars->mvars->work, info, vars->mvars);   

    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->mvars->bvec, 0.0, vars->mvars->work);
   
    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->mat_main, vars->w, 0.0, vars->mvars->work1);

    gsl_vector_sub(vars->mvars->work, vars->mvars->work1);

    gsl_blas_dgemv (CblasNoTrans, 1.0, vars->mvars->mat_main, vars->mvars->work, 0.0, vars->work);
    //double stepsize = 0.001*sqrt(vars->mvars->t);
    //stepsize = stepsize>0.01?0.01:stepsize;
    double stepsize = params->alpha_t;
    //double deltaw = 10000;
    for(int i = 0; i < 1; i++){
    //while(deltaw > 10){
       //printf("delta w is %f\n", deltaw);
       //gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->mat_main, vars->w, 0.0, vars->mvars->work1);
       //gsl_blas_dgemv (CblasNoTrans, 1.0, vars->mvars->projmat, vars->mvars->work1, 0.0, vars->work);
       /****compute inverse times vector*****/
       //gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->work_mat_main, vars->mvars->work1, 0.0, vars->mvars->work1);
       //gsl_vector_mul(vars->mvars->work1, vars->mvars->work2);
       //gsl_blas_dgemv (CblasNoTrans, 1.0, vars->mvars->delta_main, vars->mvars->work1, 0.0, vars->work);          /****end that computing********/
       //gsl_vector_sub(vars->work, vars->mvars->bvec);
       //deltaw = gsl_blas_dasum(vars->work);
       //gsl_blas_daxpy(0.1, vars->w, vars->work);
       gsl_blas_daxpy (stepsize, vars->work, vars->w);
    //}
    }
    return 1;
}

int B_PAPw(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;
    if (vars->mvars->t == 0) {
        generate_random_matrix(vars->mvars->projmat, 1.0/sqrt(vars->mvars->projmat->size2), 0, vars->mvars->rt);
        vars->w->size = vars->mvars->r;
    }
    vars->update_trace(vars->e, params, info);
    update_bvec(vars->e, info, vars->mvars);
    compute_dvec(vars->mvars->dvec, info);
    //double delta = compute_delta(vars->w, info);

    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->e, 0.0, vars->mvars->work);

    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->mvars->dvec, 0.0, vars->mvars->work1);
    vars->update_mat(vars->mvars->work, vars->mvars->work1, info, vars->mvars);
    gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->mvars->bvec, 0.0, vars->mvars->work);

    gsl_blas_dgemv (CblasNoTrans, 1.0, vars->mvars->mat_main, vars->w, 0.0, vars->mvars->work1);

    gsl_vector_sub(vars->mvars->work, vars->mvars->work1);

    double stepsize = params->alpha_t;
    for(int i = 0; i < 10; i++){
      gsl_blas_daxpy (stepsize, vars->mvars->work, vars->w);
    }
    return 1;
}

int B_Aw(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;
    if (vars->mvars->t == 0) {
        generate_random_matrix(vars->mvars->projmat, 1.0/sqrt(vars->mvars->projmat->size2), 0, vars->mvars->rt);
    }
  
    vars->update_trace(vars->e, params, info);

    update_bvec(vars->e, info, vars->mvars);

    compute_dvec(vars->mvars->dvec, info);
    //gsl_vector* temp = gsl_vector_alloc(vars->mvars->r);
    //gsl_blas_dgemv (CblasTrans, 1.0, vars->mvars->projmat, vars->e, 0.0, temp);
    //gsl_blas_dgemv (CblasNoTrans, 1.0, vars->mvars->projmat, temp, 0.0, vars->work);
    //gsl_vector_free(temp);
    double delta = compute_delta(vars->w, info);

    vars->update_mat(vars->e, vars->mvars->dvec, info, vars->mvars);

    //gsl_blas_dgemv (CblasNoTrans, 1.0, vars->mvars->mat_main, vars->w, 0.0, vars->work);

    //gsl_vector_sub(vars->work, vars->mvars->bvec);

    double stepsize = params->alpha_t;
    for(int i = 0; i < 5; i++){
      gsl_blas_dgemv (CblasNoTrans, 1.0, vars->mvars->mat_main, vars->w, 0.0, vars->work);
      gsl_vector_sub(vars->work, vars->mvars->bvec);
      gsl_blas_daxpy (-stepsize, vars->work, vars->w);
    }

    return 1;
}

int LSTD_unit(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;

    double evec = info->gamma_t*params->lambda_t*gsl_vector_get(vars->e, 0) + 1.0;
    gsl_vector_set(vars->e, 0, evec);

    vars->bvec = vars->bvec + info->reward*gsl_vector_get(vars->e, 0);

    vars->amat += gsl_vector_get(vars->e, 0)*(1.0 - info->gamma_tp1*1.0);

    if(vars->amat != 0)
        gsl_vector_set(vars->w, 0, vars->bvec*1.0/vars->amat);
    return 1;
}

void compute_values_matrix(gsl_vector * values, const gsl_matrix * observations, const void * alg_vars, const char*name, const int sparse) {
    struct matrix_alg_vars_t * vars = (struct matrix_alg_vars_t *) alg_vars;
    if(strcmp(name, "LSTD-Unit") == 0){
       gsl_vector_set_all(values, gsl_vector_get(vars->w, 0));
       return;
    }
    else if(strcmp(name, "B-PAPw") == 0){
       vars->w->size = vars->mvars->r;
       gsl_blas_dgemv (CblasNoTrans, 1.0, vars->mvars->projmat, vars->w, 0.0, vars->work);
       if(sparse == SPARSE){
         gsl_blas_dgespmv(observations, vars->work, values);
       }
       else gsl_blas_dgemv (CblasNoTrans, 1.0, observations, vars->work, 0.0, values); 
       return;
    }   
    if(sparse == SPARSE){
       gsl_blas_dgespmv(observations, vars->w, values); 
       return;
    }
    gsl_blas_dgemv (CblasNoTrans, 1.0, observations, vars->w, 0, values);
}
