functions {
  #include tr_model_functions.stanfunctions
}

data {
  int N;
  int n_meals;
  int n_train_meals;
  int num_nutrients;

  int num_ind;
  array[num_ind] int num_gluc_ind;
  array[num_ind] int num_meals_ind;
  array[num_ind] int num_train_gluc_ind;
  array[num_ind] int num_train_meals_ind;
  array[num_ind, N] int ind_idx_gluc;
  array[num_ind, n_meals] int ind_idx_meals;
  array[num_ind, n_train_meals] int ind_idx_train_meals;

  vector[N] time;
  vector[N] glucose_train;
  vector[n_meals] meal_timing;
  matrix[n_meals, num_nutrients] nutrients;
}

transformed data {
  real epsilon = 1e-3;
  int max_times = max(num_gluc_ind);
}

parameters {
  vector[num_nutrients] response_magnitude_hier_means;
  vector<lower=0>[num_nutrients] response_magnitude_hier_std;
  array[num_ind] real<lower=0> base;
  array[num_ind] real<lower=0> sigma;
  array[num_ind] vector[num_nutrients] response_magnitude_params_raw;
  array[num_ind] vector[num_nutrients] response_length_params;
  array[num_ind] real<lower=0> response_const;
  array[num_ind] real<lower=0> meal_reporting_noise;
  array[num_ind] real meal_reporting_bias;
  vector[n_train_meals] meal_timing_eiv_raw;
  array[num_ind] real<lower=0, upper=1> beta;
  array[num_ind] real theta;
  #vector<lower=0, upper=1>[n_train_meals]  err_param;
}

transformed parameters {
  vector[n_train_meals] meal_timing_eiv =meal_timing[:n_train_meals] +  meal_timing_eiv_raw;
  array[num_ind] vector[num_nutrients] response_magnitude_params;

  for (i in 1:num_ind) {
  response_magnitude_params[i] = response_magnitude_params_raw[i] .* response_magnitude_hier_std + response_magnitude_hier_means;
  }
}

generated quantities {
  vector[N] glucose;
  vector[N] clean_response;
  vector[N] trend;
  vector[N] total_response;
  vector[N] SAP;
  vector[n_meals]	rep_meal_response_timings;
  vector[n_meals] rep_meal_response_lenghts;
  vector[n_meals] rep_meal_response_magnitudes;

  for (i in 1:num_ind) {
  //likelihoods for each individual
    int ind_gluc = num_gluc_ind[i];
    int ind_meal = num_meals_ind[i];
    int ind_train_meal = num_train_meals_ind[i];
    array[ind_gluc] int gluc_selector = ind_idx_gluc[i][:num_gluc_ind[i]];
    array[ind_meal] int meal_selector = ind_idx_meals[i][:num_meals_ind[i]];
    array[ind_train_meal] int train_meal_selector = ind_idx_train_meals[i][:num_train_meals_ind[i]];
    vector[ind_gluc] ind_time = time[gluc_selector];
    vector[ind_meal] ind_meal_eiv;
    {
      ind_meal_eiv[:ind_train_meal] = meal_timing_eiv[train_meal_selector];
      for (m in ind_train_meal+1:ind_meal){
        ind_meal_eiv[m] = normal_rng(meal_timing[meal_selector][m] - meal_reporting_bias[i] ,meal_reporting_noise[i]);
      }
	    rep_meal_response_timings[meal_selector] = ind_meal_eiv;
    }
    matrix[ind_meal, num_nutrients] ind_nutr = nutrients[meal_selector];

    vector[ind_meal] meal_response_magnitudes = (1-inv_logit(ind_nutr[,4:] * response_magnitude_params[i][4:])) .* (ind_nutr[,:3] * response_magnitude_params[i][:3]);
    #meal_response_magnitudes[:ind_train_meal] =err_param[train_meal_selector] .* meal_response_magnitudes[:ind_train_meal];

    vector[ind_meal] meal_response_lengths = ind_nutr * response_length_params[i] + response_const[i];

    rep_meal_response_lenghts[meal_selector] = meal_response_lengths;
    rep_meal_response_magnitudes[meal_selector] = meal_response_magnitudes;
    vector[ind_gluc] mu = response(ind_gluc, ind_meal, ind_time, ind_meal_eiv, meal_response_magnitudes, meal_response_lengths, 0);

    int trs = num_train_gluc_ind[i];
    vector[ind_gluc] ystar = glucose_train[gluc_selector]-mu-base[i];
    vector[ind_gluc] gp_mu;
    
    vector[ind_gluc] err;
    err[1] = ystar[1];
    for (t in 2:trs) {
      err[t] = ystar[t] - (beta[i] * ystar[t-1] + theta[i] * err[t-1]);
    }
    for (t in trs+1:ind_gluc) {
      err[t] = normal_rng(0, sigma[i]);
    }
    gp_mu[1] = base[i];
    gp_mu[2:trs+1] = base[i] + ystar[:trs] * beta[i] + err[:trs] * theta[i];
    for (t in trs+2:ind_gluc) {
      gp_mu[t] = base[i] + (gp_mu[t-1] - base[i]) * beta[i]+ err[t-1] * theta[i];
    }



    trend[gluc_selector] = gp_mu;
    clean_response[gluc_selector] = gp_mu + mu;
    total_response[gluc_selector] = mu;
    glucose[gluc_selector] = to_vector(normal_rng(clean_response[gluc_selector], sigma[i]));
    for (j in 1:ind_gluc) {
      SAP[gluc_selector[j]] = normal_lpdf(glucose_train[gluc_selector[j]]| glucose[gluc_selector[j]], sigma[i]);
    }
  }

}
