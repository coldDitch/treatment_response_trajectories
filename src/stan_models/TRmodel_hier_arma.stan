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
  array[num_ind] real<lower=0> base;
  array[num_ind] real<lower=0, upper=5> sigma;
  array[num_ind] vector<lower=0>[num_nutrients] response_magnitude_params_raw;
  array[num_ind] vector[num_nutrients] response_length_params;
  array[num_ind] real<lower=0> response_const;
  array[num_ind] real<lower=0> meal_reporting_noise;
  array[num_ind] real meal_reporting_bias;
  vector[n_meals] meal_timing_eiv_raw;
  array[num_ind] real<lower=0, upper=1> beta;
  array[num_ind] real<lower=-1, upper=1> theta;
  #vector<lower=0, upper=1>[n_meals]  err_param;
}
transformed parameters {
  vector[n_meals] meal_timing_eiv = meal_timing + meal_timing_eiv_raw;
  array[num_ind] vector[num_nutrients] response_magnitude_params;

  for (i in 1:num_ind) {
  response_magnitude_params[i] = response_magnitude_params_raw[i] .* response_magnitude_hier_std + response_magnitude_hier_means;

  }
  
}


model {
  meal_timing_eiv_raw ~ normal(0, 100);
  response_magnitude_hier_means ~ normal(0, 5);
  response_magnitude_hier_std ~ normal(0, 1);
  #err_param ~ beta(10, 1);


  for (i in 1:num_ind) {
    int ind_gluc = num_gluc_ind[i];
    int ind_meal = num_meals_ind[i];
    array[ind_gluc] int gluc_selector = ind_idx_gluc[i][:num_gluc_ind[i]];
    array[ind_meal] int meal_selector = ind_idx_meals[i][:num_meals_ind[i]];
    vector[ind_gluc] ind_time = time[gluc_selector];
    vector[ind_meal] ind_meal_eiv = meal_timing_eiv[meal_selector];
    matrix[ind_meal, num_nutrients] ind_nutr = nutrients[meal_selector];
    vector[ind_meal] meal_response_magnitudes = (1-inv_logit(ind_nutr[,4:] * response_magnitude_params[i][4:])) .* (ind_nutr[,:3] * response_magnitude_params[i][:3]);
    vector[ind_meal] meal_response_lengths = ind_nutr * response_length_params[i] + response_const[i];
    vector[ind_gluc] mu = response(ind_gluc, ind_meal, ind_time, ind_meal_eiv, meal_response_magnitudes, meal_response_lengths, 0);
    

    vector[ind_gluc] ystar = glucose[gluc_selector] - mu - base[i];
    vector[ind_gluc] err;
    err[1] = ystar[1];
    for (t in 2:ind_gluc) {
      err[t] = ystar[t] - (beta[i] * ystar[t-1] + theta[i] * err[t-1]);
    }

    //priors
    sigma[i] ~ normal(0, 1);
    base[i] ~ normal(5, 1);
    response_magnitude_params_raw[i] ~ std_normal();
    response_length_params[i] ~ normal(0, 1);
    response_const[i] ~ normal(20,5);

    meal_reporting_noise[i] ~ normal(0, 100);
    meal_reporting_bias[i] ~ normal(0, 100);

    if (use_prior){
      for (j in 1:ind_meal) {
        meal_response_magnitudes[j] ~ inv_gamma(1.5, 5);
      }
    }

  //likelihoods for each individual
    ystar[2:ind_gluc] ~ normal(beta[i] * ystar[:ind_gluc-1] + theta[i] * err[:ind_gluc-1], sigma[i]);

    meal_timing[meal_selector] ~ normal(meal_timing_eiv[meal_selector] + meal_reporting_bias[i], meal_reporting_noise[i]);
  }
}
