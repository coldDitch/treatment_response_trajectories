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
  real marg_std;
  real lengthscale;
}

transformed data {
  real epsilon = 1e-3;
  int max_times = max(num_gluc_ind);
}

parameters {
  vector[num_nutrients] response_magnitude_hier_means;
  vector<lower=0>[num_nutrients] response_magnitude_hier_std;
  real<lower=0> response_const_means;
  real<lower=0> response_const_std;
  array[num_ind] real<lower=0> base;
  array[num_ind] real<lower=0> sigma;
  array[num_ind] vector[num_nutrients] response_magnitude_params_raw;
  array[num_ind] real<lower=0> response_const_raw;
  array[num_ind] real<lower=0> meal_reporting_noise;
  array[num_ind] real meal_reporting_bias;
  vector[n_train_meals] meal_timing_eiv_raw;
}

transformed parameters {
  vector[n_train_meals] meal_timing_eiv = meal_timing_eiv_raw + meal_timing[:n_train_meals];
  array[num_ind] real<lower=0> response_const;
  array[num_ind] vector[num_nutrients] response_magnitude_params;

  for (i in 1:num_ind) {
  response_magnitude_params[i] = response_magnitude_params_raw[i] .* response_magnitude_hier_std + response_magnitude_hier_means;
  response_const[i] = response_const_raw[i] * response_const_std + response_const_means;
  }
}

generated quantities {
  vector[N] glucose;
  vector[N] clean_response;
  vector[N] trend;
  vector[N] total_response;
  vector[n_train_meals]	rep_meal_response_timings;
  vector[n_train_meals] rep_meal_response_lenghts;
  vector[n_train_meals] rep_meal_response_magnitudes;

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
	    rep_meal_response_timings[ind_idx_train_meals[i][:num_train_meals_ind[i]]] = meal_timing_eiv[ind_idx_train_meals[i][:num_train_meals_ind[i]]];
      for (m in ind_train_meal+1:ind_meal){
        ind_meal_eiv[m] = normal_rng(meal_timing[meal_selector][m] + meal_reporting_bias[i] ,meal_reporting_noise[i]);
      }
    }
    matrix[ind_meal, num_nutrients] ind_nutr = nutrients[meal_selector];

    vector[ind_meal] meal_response_magnitudes = (1-inv_logit(ind_nutr[,4:] * response_magnitude_params[i][4:])) .* (ind_nutr[,:3] * response_magnitude_params[i][:3]);

    vector[ind_meal] meal_response_lengths = rep_vector(response_const[i], ind_meal);

    rep_meal_response_lenghts[ind_idx_train_meals[i][:num_train_meals_ind[i]]] = meal_response_lengths[:num_train_meals_ind[i]];
    rep_meal_response_magnitudes[ind_idx_train_meals[i][:num_train_meals_ind[i]]] = meal_response_magnitudes[:num_train_meals_ind[i]];
    vector[ind_gluc] mu = response(ind_gluc, ind_meal, ind_time, ind_meal_eiv, meal_response_magnitudes, meal_response_lengths, base[i]);

    int trs = num_train_gluc_ind[i];
    vector[ind_gluc] gp_mu = draw_pred_rng(to_array_1d(ind_time),
                       glucose_train[gluc_selector][:trs],
                       mu[:trs],
                       to_array_1d(ind_time[:trs]),
                       marg_std,
                       lengthscale,
                       epsilon,
                       mu,
                       base[i],
                       sigma[i])-base[i];

    trend[gluc_selector] = gp_mu + base[i];
    clean_response[gluc_selector] = gp_mu + mu;
    total_response[gluc_selector] = mu - base[i];
    for (g in 1:ind_gluc){
      glucose[gluc_selector[g]] = normal_rng((gp_mu + mu)[g], sigma[i]);
    }


  }
}
