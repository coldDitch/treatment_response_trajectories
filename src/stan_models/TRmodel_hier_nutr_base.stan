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
}

transformed data {
  real epsilon = 1e-3;
}

parameters {
  vector[num_nutrients] response_magnitude_hier_means;
  vector<lower=0>[num_nutrients] response_magnitude_hier_std;
  real<lower=0> response_const_means;
  real<lower=0> response_const_std;
  array[num_ind] real<lower=0> base;
  array[num_ind] real<lower=0> lengthscale;
  array[num_ind] real<lower=0> marg_std;
  array[num_ind] real<lower=0> sigma;
  array[num_ind] vector[num_nutrients] response_magnitude_params;
  array[num_ind] real<lower=0> response_const;
  array[num_ind] real<lower=0> meal_reporting_noise;
  vector[n_meals] meal_timing_eiv;
}


model {
  response_magnitude_hier_means ~ normal(0, 5);
  response_magnitude_hier_std ~ normal(0, 0.5);
  response_const_means ~ normal(0, 5);
  response_const_std ~ normal(0, 1);



  for (i in 1:num_ind) {

    //priors
    lengthscale[i] - 10 ~ normal(0, 10);
    sigma[i] ~ normal(0, 10);
    marg_std[i] ~ normal(0, 0.1);
    base[i] ~ normal(4, 1);
    response_magnitude_params[i] ~ normal(response_magnitude_hier_means, response_magnitude_hier_std);
	response_const[i] ~ normal(response_const_means, response_const_std);

  //likelihoods for each individual
    int ind_gluc = num_gluc_ind[i];
    int ind_meal = num_meals_ind[i];
    array[ind_gluc] int gluc_selector = ind_idx_gluc[i][:num_gluc_ind[i]];
    array[ind_meal] int meal_selector = ind_idx_meals[i][:num_meals_ind[i]];
    vector[ind_gluc] ind_time = time[gluc_selector];
    vector[ind_meal] ind_meal_eiv = meal_timing_eiv[meal_selector];
    matrix[ind_meal, num_nutrients] ind_nutr = nutrients[meal_selector];
    vector[ind_meal] meal_response_magnitudes = (1-inv_logit(ind_nutr[,4:] * response_magnitude_params[i][4:])) .* (ind_nutr[,:3] * response_magnitude_params[i][:3]);
    vector[ind_meal] meal_response_lengths = rep_vector(response_const[i], ind_meal);
    vector[ind_gluc] mu = response(ind_gluc, ind_meal, ind_time, ind_meal_eiv, meal_response_magnitudes, meal_response_lengths, base[i]);
	
	matrix[ind_gluc, ind_gluc] K = gp_exp_quad_cov(to_array_1d(ind_time), marg_std[i], lengthscale[i]);
	matrix[ind_gluc, ind_gluc] L_K;
	for (j in 1:ind_meal) {
		K[j,j] = K[j,j] + sigma[i];
	}
	L_K = cholesky_decompose(K);

    if (use_prior){
      for (j in 1:ind_meal) {
        meal_response_magnitudes[j] ~ normal(1, 1);
      }
    }

    meal_reporting_noise[i] ~ normal(0, 10);


    glucose[gluc_selector] ~ multi_normal_cholesky(mu, L_K);


    meal_timing[meal_selector] ~ normal(meal_timing_eiv[meal_selector], meal_reporting_noise[i]);
  }
}
