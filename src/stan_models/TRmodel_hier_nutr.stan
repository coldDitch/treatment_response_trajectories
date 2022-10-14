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
  vector<lower=0>[num_nutrients] response_magnitude_hier_means;
  vector<lower=0>[num_nutrients] response_magnitude_hier_std;
  vector<lower=0>[num_nutrients] response_length_hier_means;
  vector<lower=0>[num_nutrients] response_length_hier_std;
  array[num_ind] vector[M] beta_GP;
  array[num_ind] real<lower=0> lengthscale;
  array[num_ind] real<lower=0> marg_std;
  array[num_ind] real<lower=0> sigma;
  array[num_ind] real<lower=0> base;
  array[num_ind] vector<lower=0>[num_nutrients] response_magnitude_params;
  array[num_ind] vector<lower=0>[num_nutrients] response_length_params;
  array[num_ind] real<lower=0> meal_reporting_noise;
  vector[n_meals] meal_timing_eiv;
}


model {
  response_length_hier_means ~ normal(0, 1);
  response_length_hier_std ~ normal(0.2, 0.2);
  response_magnitude_hier_means ~ normal(0, 10);
  response_magnitude_hier_std ~ normal(0.2, 0.2);



  for (i in 1:num_ind) {

    //priors
    beta_GP[i] ~ normal(0, 1);
    lengthscale[i] ~ inv_gamma(1, 3);
    sigma[i] ~ normal(0, 1);
    marg_std[i] ~ normal(0.2, 0.2);
    base[i] ~ normal(4, 1);
    response_magnitude_params[i] ~ normal(response_magnitude_hier_means, response_magnitude_hier_std);
    response_length_params[i] ~ normal(response_length_hier_means, response_length_hier_std);

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
    vector[ind_meal] meal_response_lengths = ind_nutr * response_length_params[i];
    vector[ind_gluc] mu = response(ind_gluc, ind_meal, ind_time, ind_meal_eiv, meal_response_magnitudes, meal_response_lengths, base[i]);

    for(m in 1:M){
      diagSPD[m] = sqrt(spd(marg_std[i], lengthscale[i], sqrt(lambda(L, m))));
    }

    SPD_beta = diagSPD .* beta_GP[i];

    gp_mu = PHI[i][:num_gluc_ind[i]] * SPD_beta;

    if (use_prior){
      for (j in 1:ind_meal) {
        meal_response_magnitudes[j] ~ inv_gamma(1, 3);
      }
    }

    meal_reporting_noise[i] ~ normal(0, 0.25);


    glucose[gluc_selector] ~ normal(gp_mu + mu, sigma[i]);


    meal_timing[meal_selector] ~ normal(meal_timing_eiv[meal_selector], meal_reporting_noise[i]);
  }
}
