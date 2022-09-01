functions {
  #include tr_model_functions.stanfunctions
}

data {

  int M;
  real L;

  int N;
  int n_meals;
  int n_train_meals;
  int num_nutrients;

  int num_ind;
  array[num_ind] int num_gluc_ind;
  array[num_ind] int num_meals_ind;
  array[num_ind] int num_train_meals_ind;
  array[num_ind, N] int ind_idx_gluc;
  array[num_ind, n_meals] int ind_idx_meals;
  array[num_ind, n_train_meals] int ind_idx_train_meals;

  vector[N] time;
  vector[n_meals] meal_timing;
  matrix[n_meals, num_nutrients] nutrients;
}

transformed data {
  real epsilon = 1e-3;
  array[num_ind] matrix[N, M] PHI;
  for (i in 1:num_ind) {
    for (m in 1:M) {
      PHI[i][:num_gluc_ind[i],m] = phi(L, m, time[ind_idx_gluc[i][:num_gluc_ind[i]]]);
    }
  }
}

parameters {
  array[num_ind] vector[M] beta_GP;
  array[num_ind] real<lower=0> lengthscale;
  array[num_ind] real<lower=0> marg_std;
  array[num_ind] real<lower=0> sigma;
  array[num_ind] real<lower=0> base;
  array[num_ind] vector[num_nutrients] response_magnitude_params;
  array[num_ind] vector[num_nutrients] response_length_params;
  array[num_ind] real<lower=0> meal_reporting_noise;
  vector[n_train_meals] meal_timing_eiv;
}


generated quantities {
  vector[N] glucose;
  vector[N] clean_response;
  vector[N] trend;
  vector[N] total_response;

  for (i in 1:num_ind) {
  //likelihoods for each individual
    int ind_gluc = num_gluc_ind[i];
    int ind_meal = num_meals_ind[i];
    int ind_train_meal = num_train_meals_ind[i];
    array[ind_gluc] int gluc_selector = ind_idx_gluc[i][:num_gluc_ind[i]];
    array[ind_meal] int meal_selector = ind_idx_meals[i][:num_meals_ind[i]];
    vector[ind_gluc] ind_time = time[gluc_selector];
    vector[ind_meal] ind_meal_eiv;
    {
      ind_meal_eiv[:ind_train_meal] = meal_timing_eiv[ind_idx_train_meals[i][:num_train_meals_ind[i]]];
      for (m in ind_train_meal+1:ind_meal){
        ind_meal_eiv[m] = normal_rng(meal_timing[meal_selector][m],meal_reporting_noise[i]);
      }
    }
    matrix[ind_meal, num_nutrients] ind_nutr = nutrients[meal_selector];
    vector[ind_gluc] gp_mu;
    vector[M] diagSPD;
    vector[M] SPD_beta;
    vector[ind_meal] meal_response_magnitudes = ind_nutr * response_magnitude_params[i];
    vector[ind_meal] meal_response_lengths = ind_nutr * response_length_params[i];
    vector[ind_gluc] mu = response(ind_gluc, ind_meal, ind_time, ind_meal_eiv, meal_response_magnitudes, meal_response_lengths, base[i]);
    
    for(m in 1:M){ 
      diagSPD[m] = sqrt(spd(marg_std[i], lengthscale[i], sqrt(lambda(L, m)))); 
    }
    
    SPD_beta = diagSPD .* beta_GP[i];
    
    gp_mu = PHI[i][:num_gluc_ind[i]] * SPD_beta;

    trend[gluc_selector] = gp_mu + base[i];
    clean_response[gluc_selector] = gp_mu + mu;
    total_response[gluc_selector] = mu - base[i];
    for (g in 1:ind_gluc){
      glucose[gluc_selector[g]] = normal_rng((gp_mu + mu)[g], sigma[i]);
    }


  }
}
