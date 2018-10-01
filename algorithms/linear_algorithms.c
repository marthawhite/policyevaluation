#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "linear_algorithms.h"

// TODO: put in correct traces, and decide what to do about trace update fcns
const static struct{
      const char *name;
      AlgUpdateFcn update_fcn;
      TraceUpdateFcn update_trace;
} list_linear_algorithms [] = {
      { "TD", TD_lambda, update_trace_replacing},
      { "TD-Unit", TD_unit, update_trace_replacing},
      { "TO-TD", TO_TD_lambda, update_trace_trueonline},
      { "ETD", ETD, update_trace_replacing},
      { "TO-ETD", TO_ETD, update_trace_replacing},
      { "GTD", GTD_lambda, update_trace_accumulating},
      { "TO-GTD", TO_GTD_lambda, update_trace_trueonline},
};
const static int num_totalalgs = (sizeof(list_linear_algorithms) / sizeof(list_linear_algorithms[0]));

void init_linear_alg(struct alg_t * alg, const int numobservations){

      // alg_vars is a void *, initialized here to the correct struct size
      alg->alg_vars = malloc(sizeof(struct linear_alg_vars_t));
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg->alg_vars;

      // TODO: different way to give numfeatures, right now assuming MDP giving observations = features
      int numfeatures = numobservations;
      
      vars->w = gsl_vector_calloc(numfeatures);
      vars->e = gsl_vector_calloc(numfeatures);

      int i;
      // Get the update functions for this algorithm
      for (i = 0; i < num_totalalgs; i++) {
         if (!strcmp(list_linear_algorithms[i].name, alg->name)) {
            alg->update_fcn = list_linear_algorithms[i].update_fcn;
            alg->reset_fcn = reset_linear_alg;
            vars->update_trace = list_linear_algorithms[i].update_trace;
            break;
         }
      }
      // All algorithm use a dot product to compute the value function
      alg->get_values = compute_values_linear;

      vars->eh = NULL;
      vars->h = NULL;
      vars->w_tm1 = NULL;
      vars->work = NULL;
      vars->work1 = NULL;
      vars->t = 0;
      vars->VofS = 0; 
      /*
       * Algorithm specific initializations
       */
      if (strcmp(alg->name, "TO-ETD") == 0) {
         vars->work = gsl_vector_alloc(numfeatures);
         vars->I = 1;
      }
      else if(strcmp(alg->name, "GTD") == 0){
         vars->h = gsl_vector_alloc(numfeatures);
         vars->work = gsl_vector_alloc(numfeatures);
      }
      else if(strcmp(alg->name, "TO-GTD") == 0){
         vars->h = gsl_vector_alloc(numfeatures);
         vars->eh = gsl_vector_alloc(numfeatures);
         vars->w_tm1 = gsl_vector_alloc(numfeatures);
         vars->work = gsl_vector_alloc(numfeatures);
         vars->work1 = gsl_vector_alloc(numfeatures);
      }
}

void deallocate_linear_alg(void * alg_vars){
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg_vars;
      
      gsl_vector_free(vars->w);
      gsl_vector_free(vars->e);
   
      if (vars->work != NULL) {
         gsl_vector_free(vars->work);
      }
      if (vars->work1 != NULL) {
         gsl_vector_free(vars->work1);
      }
      if (vars->h != NULL) {
         gsl_vector_free(vars->h);
      }
      if (vars->eh != NULL) {
         gsl_vector_free(vars->eh);
      }
      if (vars->w_tm1 != NULL) {
         gsl_vector_free(vars->w_tm1);
      }

      free(vars);
}

void reset_linear_alg(void * alg_vars){
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg_vars;

      gsl_vector_set_zero(vars->w);
      gsl_vector_set_zero(vars->e);

      vars->t = 0;

      vars->I = 1;
      vars->M = 0;
      vars->F = 0;
      vars->D = 0;

      if (vars->h != NULL) {
         gsl_vector_set_zero(vars->h);
      }
      if (vars->eh != NULL) {
         gsl_vector_set_zero(vars->eh);
      }
      if (vars->w_tm1 != NULL) {
         gsl_vector_set_zero(vars->w_tm1);
      }
      if (vars->work != NULL) {
         gsl_vector_set_zero(vars->work);
      }
      if (vars->work1 != NULL) {
         gsl_vector_set_zero(vars->work1);
      } 
      // TODO: ensure ETD and TO-TD have correct reinitializations of VofS, etc.
      // Reset happens at the beginning at each run, so do not need to consider ETDs vars here
      // VofS, etc will be handled inside the algorithms
}

int TD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg_vars;
      double delta = compute_delta(vars->w, info);
      vars->update_trace(vars->e, params, info);
      gsl_blas_daxpy (params->alpha_t*delta, vars->e, vars->w);
      //printf("gamma is %f reward is %f\n", info->gamma_t, info->reward);
      //gsl_vector_view subx = gsl_vector_subvector(info->x_t, 0, 20);
      //gsl_vector_print(&subx.vector); 
      return 0;
}


/* Store current value in VofS, because need that value for the next step */
int TO_TD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info) {
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg_vars;

      double new_v = 0;
      gsl_blas_ddot(vars->w, info->x_tp1, &new_v);  
      double delta = info->reward + info->gamma_tp1*new_v - vars->VofS;
      // printf("the current delta is %f\n", delta);
      // Update trace; ignores vars->update_trace, because must use true-online update
      update_trace_trueonline(vars->e, params, info);

      // Update weights
      double dot = 0;
      gsl_blas_ddot(vars->w, info->x_t, &dot);    
      // TODO: include function in info to compute sparse gsl_blas_daxpy
      gsl_blas_daxpy (delta, vars->e, vars->w);
      gsl_blas_daxpy (params->alpha_t*(vars->VofS - dot), info->x_t, vars->w);

      // Save VofS for next step
      vars->VofS = new_v;
    
      return 0;
}

int ETD(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info) {
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg_vars;
     
    vars->F = info->rho_t*info->gamma_t*(vars->F) + vars->I;
    
    double M = params->lambda_t*vars->I + (1 - params->lambda_t)*(vars->F);
    
    gsl_vector_scale(vars->e, info->gamma_t*info->rho_t*params->lambda_t);
    
    gsl_blas_daxpy(M*info->rho_t, info->x_t, vars->e);

    double delta = compute_delta(vars->w, info);

    gsl_blas_daxpy(params->alpha_t*delta, vars->e, vars->w);

    return 0;
}

int TO_ETD(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info) {
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg_vars;
      
      double delta = compute_delta(vars->w, info);
    
      vars->F = vars->F*info->rho_tm1*info->gamma_t + vars->I;
    
      // update eligibility trace
      double M = params->lambda_t*vars->I + (1-params->lambda_t)*vars->F;
      double phie = 0;
      gsl_blas_ddot (info->x_t, vars->e, &phie);
    
      double S = info->rho_t*params->alpha_t* M * (1 - info->rho_t*info->gamma_t*params->lambda_t*phie);
      gsl_vector_scale(vars->e,info->rho_t*info->gamma_t*params->lambda_t);
      gsl_blas_daxpy (S, info->x_t, vars->e);
    
      /* Rich's approach */
      gsl_vector_memcpy(vars->work, info->x_t);
      gsl_vector_scale(vars->work, -params->alpha_t*M*info->rho_t);
      gsl_vector_add(vars->work, vars->e);
      gsl_vector_scale(vars->work, vars->D);
      gsl_blas_daxpy (delta, vars->e, vars->work);
      gsl_vector_add(vars->w, vars->work);
    
      gsl_blas_ddot (vars->work, info->x_tp1, &(vars->D));
    
      return 0;
}

int GTD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg_vars;
    
      double delta = compute_delta(vars->w, info);
    
      vars->update_trace(vars->e, params, info);
    
      gsl_vector_scale(vars->e, info->rho_t);
    
      gsl_blas_daxpy (params->alpha_t*delta, vars->e, vars->w);
    
      double eh = 0;
      gsl_blas_ddot (vars->e, vars->h, &eh);
    
      double glew = -params->alpha_t*info->gamma_tp1*(1-params->lambda_tp1)*eh;
      gsl_blas_daxpy (glew, info->x_tp1, vars->w);
    
      eh = 0;
      gsl_blas_ddot (info->x_t, vars->h, &eh);
      gsl_blas_daxpy (params->beta_t*delta, vars->e, vars->h);
      gsl_blas_daxpy (-eh*params->beta_t, info->x_t, vars->h);
    
      return 0;
}

int TO_GTD_lambda(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info) {
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg_vars;
    
      //the work matrix here recorded the old w
      gsl_vector_memcpy(vars->work, vars->w);
    
      double delta = compute_delta(vars->w, info);
    
      update_trace_to_gtd(vars->e, params, info);
    
      double dot=0;
      gsl_blas_ddot (vars->eh, info->x_t, &dot);
      gsl_vector_scale(vars->eh,info->gamma_t*params->lambda_t*info->rho_tm1);
      gsl_blas_daxpy (params->beta_t*(1 - info->rho_tm1*info->gamma_t*params->lambda_t*dot), info->x_t, vars->eh);
    
      //work1 vector will store the point-wise dif between w and w_tm1
      gsl_vector_memcpy(vars->work1, vars->w);
      gsl_vector_sub(vars->work1, vars->w_tm1);
      dot = 0;
      gsl_blas_ddot (vars->work1, info->x_t, &dot);
      gsl_vector_memcpy(vars->work1, info->x_t);
      gsl_vector_scale(vars->work1,(-params->alpha_t*info->rho_t));
      gsl_vector_add(vars->work1, vars->e);
      gsl_blas_daxpy (dot, vars->work1, vars->w);
    
      gsl_blas_daxpy (delta, vars->e, vars->w);
    
      dot = 0;
      gsl_blas_ddot (vars->h, vars->e, &dot);
      dot = params->alpha_t*info->gamma_tp1*(1.0-params->lambda_tp1)*dot;
      gsl_blas_daxpy (-dot, info->x_tp1, vars->w);
    
      dot = 0;
      gsl_blas_ddot (vars->h, info->x_t, &dot);
      gsl_blas_daxpy (info->rho_t*delta, vars->eh, vars->h);
      gsl_blas_daxpy (-params->beta_t*dot, info->x_t, vars->h);
    
      gsl_vector_memcpy(vars->w_tm1, vars->work);
    
      return 0;
}

int TD_unit(void * alg_vars, const struct alg_params_t * params, const struct transition_info_t * info){
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg_vars;
      double delta = info->reward + info->gamma_tp1*gsl_vector_get(vars->w, 0) - gsl_vector_get(vars->w, 0);
      double evec = info->gamma_t*params->lambda_t*gsl_vector_get(vars->e, 0) + 1.0;
      gsl_vector_set(vars->e, 0, evec);
      double wvec = gsl_vector_get(vars->w, 0)+params->alpha_t*gsl_vector_get(vars->e, 0)*delta;
      gsl_vector_set(vars->w, 0, wvec);
      return 0;
}

void compute_values_linear(gsl_vector * values, const gsl_matrix * observations, const void * alg_vars, const char*name, const int sparse) {
      struct linear_alg_vars_t * vars = (struct linear_alg_vars_t *) alg_vars;
      if(strcmp(name, "TD-Unit") == 0){
       gsl_vector_set_all(values, gsl_vector_get(vars->w, 0));
       return;
    }
    if(sparse == SPARSE){
       gsl_blas_dgespmv(observations, vars->w, values);          
       return;
    }
      gsl_blas_dgemv (CblasNoTrans, 1.0, observations, vars->w, 0, values);
}
