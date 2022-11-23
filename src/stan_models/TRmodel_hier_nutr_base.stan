functions {
  #include tr_model_functions.stanfunctions
}

data {
  int<lower=0, upper=1> use_prior;
  int N;
  int n_meals;
  int num_nutrients;

  int num_ind;
  array[num_ind] int num_gluc_ind;
  array[num_ind] int num_meals_ind;
  array[num_ind, N] int ind_idx_gluc;
  array[num_ind, n_meals] int ind_idx_meals;

  vector[N] time;
  vector[N] glucose;
  vector[n_meals] meal_timing;
  matrix[n_meals, num_nutrients] nutrients;
  real marg_std;
  real lengthscale;
}

transformed data {
  real epsilon = 1e-3;
  int max_times = max(num_gluc_ind);
  array[num_ind] matrix[max_times, max_times] covL_K;
  for (i in 1:num_ind){
    int ind_gluc = num_gluc_ind[i];
    array[ind_gluc] int gluc_selector = ind_idx_gluc[i][:num_gluc_ind[i]];
    vector[ind_gluc] ind_time = time[gluc_selector];
    matrix[ind_gluc, ind_gluc] K = gp_exp_quad_cov(to_array_1d(ind_time), marg_std, lengthscale);
    matrix[ind_gluc, ind_gluc] L_K;
    for (j in 1:ind_gluc) { 
      K[j,j] = K[j,j] + epsilon;
    }
    L_K = cholesky_decompose(K);
    covL_K[i][:ind_gluc, :ind_gluc] = L_K;
  }
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
  vector[n_meals] meal_timing_eiv_raw;
  vector[N] eta;
}
transformed parameters {
  vector[n_meals] meal_timing_eiv = meal_timing_eiv_raw + meal_timing;
  array[num_ind] real<lower=0> response_const;
  array[num_ind] vector[num_nutrients] response_magnitude_params;

  for (i in 1:num_ind) {
  response_magnitude_params[i] = response_magnitude_params_raw[i] .* response_magnitude_hier_std + response_magnitude_hier_means;
  response_const[i] = response_const_raw[i] * response_const_std + response_const_means;

  }
  
}


model {
  meal_timing_eiv_raw ~ normal(0, 100);
  response_magnitude_hier_means ~ normal(0, 5);
  response_magnitude_hier_std ~ normal(0, 0.5);
  response_const_means ~ normal(0, 5);
  response_const_std ~ normal(0, 1);



  for (i in 1:num_ind) {

    //priors
    sigma[i] ~ normal(0, 0.01);
    base[i] ~ normal(4, 1);
    response_magnitude_params_raw[i] ~ std_normal();
	  response_const[i] ~ std_normal();

  //likelihoods for each individual
    int ind_gluc = num_gluc_ind[i];
    int ind_meal = num_meals_ind[i];
    array[ind_gluc] int gluc_selector = ind_idx_gluc[i][:num_gluc_ind[i]];
    array[ind_meal] int meal_selector = ind_idx_meals[i][:num_meals_ind[i]];
    vector[ind_gluc] eta_ind = eta[gluc_selector];
    vector[ind_gluc] ind_time = time[gluc_selector];
    vector[ind_meal] ind_meal_eiv = meal_timing_eiv[meal_selector];
    matrix[ind_meal, num_nutrients] ind_nutr = nutrients[meal_selector];
    vector[ind_meal] meal_response_magnitudes = (1-inv_logit(ind_nutr[,4:] * response_magnitude_params[i][4:])) .* (ind_nutr[,:3] * response_magnitude_params[i][:3]);
    vector[ind_meal] meal_response_lengths = rep_vector(response_const[i], ind_meal);
    vector[ind_gluc] mu = response(ind_gluc, ind_meal, ind_time, ind_meal_eiv, meal_response_magnitudes, meal_response_lengths, base[i]);
/*	
	matrix[ind_gluc, ind_gluc] K = gp_exp_quad_cov(to_array_1d(ind_time), marg_std[i], lengthscale[i]);
	matrix[ind_gluc, ind_gluc] L_K;
	for (j in 1:ind_meal) {
		K[j,j] = K[j,j] + sigma[i];
	}
	L_K = cholesky_decompose(K);
*/
    if (use_prior){
      for (j in 1:ind_meal) {
        meal_response_magnitudes[j] ~ normal(1, 1);
      }
    }

    meal_reporting_noise[i] ~ normal(0, 10);
    meal_reporting_bias[i] ~ normal(0, 10);

    vector[ind_gluc] gp_mu = covL_K[i][:ind_gluc, :ind_gluc] * eta_ind;

    glucose[gluc_selector] ~ normal(mu + gp_mu, sigma[i]);


    meal_timing[meal_selector] ~ normal(meal_timing_eiv[meal_selector] + meal_reporting_bias[i], meal_reporting_noise[i]);
  }
}
