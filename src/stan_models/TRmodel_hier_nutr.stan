functions {
  #include tr_model_functions.stanfunctions
}

data {

  int<lower=0, upper=1> use_prior;
  int M;
  real L;

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
  array[num_ind] matrix[N, M] PHI;
  for (i in 1:num_ind) {
    for (m in 1:M) {
      PHI[i][:num_gluc_ind[i],m] = phi(L, m, time[ind_idx_gluc[i][:num_gluc_ind[i]]]);
    }
  }
}

parameters {
  vector[num_nutrients] response_magnitude_hier_means;
  vector<lower=0>[num_nutrients] response_magnitude_hier_std;
  real<lower=0> response_const_means;
  real<lower=0> response_const_std;

  array[num_ind] vector[M] beta_GP;
  array[num_ind] real<lower=0> lengthscale;
  array[num_ind] real<lower=0> marg_std;
  array[num_ind] real<lower=0> sigma;
  array[num_ind] real<lower=0> base;
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
    beta_GP[i] ~ normal(0, 1);
    lengthscale[i] ~ normal(0, 10);
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
    vector[ind_gluc] gp_mu;
    vector[M] diagSPD;
    vector[M] SPD_beta;
    vector[ind_meal] meal_response_magnitudes = (1-inv_logit(ind_nutr[,4:] * response_magnitude_params[i][4:])) .* (ind_nutr[,:3] * response_magnitude_params[i][:3]);
    vector[ind_meal] meal_response_lengths = rep_vector(response_const[i], ind_meal);
    vector[ind_gluc] mu = response(ind_gluc, ind_meal, ind_time, ind_meal_eiv, meal_response_magnitudes, meal_response_lengths, base[i]);

    for(m in 1:M){
      diagSPD[m] = sqrt(spd(marg_std[i], lengthscale[i], sqrt(lambda(L, m))));
    }

    SPD_beta = diagSPD .* beta_GP[i];

    gp_mu = PHI[i][:num_gluc_ind[i]] * SPD_beta;

    if (use_prior){
      for (j in 1:ind_meal) {
        meal_response_magnitudes[j] ~ normal(1, 1);
      }
    }

    meal_reporting_noise[i] ~ normal(0, 10);


    glucose[gluc_selector] ~ normal(gp_mu + mu, sigma[i]);


    meal_timing[meal_selector] ~ normal(meal_timing_eiv[meal_selector], meal_reporting_noise[i]);
  }
}
