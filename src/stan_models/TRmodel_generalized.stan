functions {
  real inverse_gaussian_lpdf(real x, real mu, real lambda) {
    return 0.5 * (-lambda *  square(mu - x)/(square(mu) * x) - 3 * log(x) - log(2 * pi()));
  }

  vector impulse(real response_magnitude, real response_length, vector delta_t) {
    return response_magnitude * exp(-0.5 * square(delta_t - 3*response_length) / square(response_length));
  }

  vector impulsev2(real response_magnitude, real response_length, vector delta_t) {
    return 0.011109 * response_magnitude * exp(delta_t .* (3*response_length - 0.5 * delta_t) / response_length ^ 2);
  }

  vector response(int N, int n_meals, vector time, vector meal_timing, vector meal_response_magnitudes, vector meal_response_lengths, real base) {
    vector[N] mu = rep_vector(base, N);
    for (i in 1:n_meals) {
        mu += impulsev2(meal_response_magnitudes[i], meal_response_lengths[i], time - meal_timing[i]);
    }
    return mu;
  }

  vector draw_pred_rng(real[] test_x,
                       vector glucose,
                       vector train_mu,
                       real[] train_x,
                       real marg_std,
                       real lengthscale,
                       real epsilon,
                       vector mu2,
                       real base,
                       real sigma) {
    int N1 = rows(glucose);
    int N2 = size(test_x);
    vector[N2] f2;
    vector[N2] f2_mu;
    {
      matrix[N1, N1] L_K;
      vector[N1] K_div_y1;
      matrix[N1, N2] k_x1_x2;
      matrix[N1, N2] v_pred;
      matrix[N2, N2] cov_f2;
      matrix[N2, N2] diag_delta;
      matrix[N1, N1] K;
      K = cov_exp_quad(train_x, marg_std, lengthscale);
      for (n in 1:N1)
        K[n, n] = K[n,n] + epsilon + sigma;
      L_K = cholesky_decompose(K);
      K_div_y1 = mdivide_left_tri_low(L_K, glucose-train_mu);
      K_div_y1 = mdivide_right_tri_low(K_div_y1', L_K)';
      k_x1_x2 = cov_exp_quad(train_x, test_x, marg_std, lengthscale);
      f2_mu = mu2 + (k_x1_x2' * K_div_y1);
      v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
      cov_f2 = cov_exp_quad(test_x, marg_std, lengthscale) - v_pred' * v_pred;
      diag_delta = diag_matrix(rep_vector(epsilon, N2));

      f2 = multi_normal_rng(f2_mu, cov_f2 + diag_delta);
    }
    return f2;
  }
}

data {
  int N;
  real time[N];
  vector[N] glucose;
  int n_meals;
  vector[n_meals] meal_timing;
  int n_pred;
  real pred_times[n_pred];
  int n_meals_pred;
  vector[n_meals_pred] pred_meals;
  int num_nutrients;
  matrix[n_meals, num_nutrients] nutrients;
  matrix[n_meals_pred, num_nutrients] pred_nutrients;
}
transformed data {
  real epsilon = 1e-3;
  real glucose_var = variance(glucose);
}

parameters {
  real<lower=epsilon> lengthscale;
  real<lower=0> marg_std;
  real<lower=0> sigma;
  real<lower=0> base;
  vector[num_nutrients] response_magnitude_params;
  vector<lower=0>[num_nutrients] response_length_params;
  real<lower=0> response_length_const;
  real<lower=0> meal_reporting_noise;
  real meal_reporting_bias;
  vector[n_meals] meal_timing_eiv;
  vector[n_meals_pred-n_meals] fut_meal_timing;
}

transformed parameters {
  vector[n_meals] meal_response_magnitudes = nutrients * response_magnitude_params/n_meals;
  vector[n_meals] meal_response_lengths = nutrients * response_length_params/n_meals + response_length_const;
  vector[N] mu = response(N, n_meals, to_vector(time), meal_timing_eiv, meal_response_magnitudes, meal_response_lengths, base);
  // todo test if pred_meals works better in generated quantities with exact inference
  vector[n_meals_pred] pred_meals_eiv;
  for (i in 1:n_meals) {
    pred_meals_eiv[i] = meal_timing_eiv[i];
  }
  for (i in n_meals+1:n_meals_pred){
    pred_meals_eiv[i] = fut_meal_timing[i-n_meals];
  }
}


model {

  //priors
  lengthscale ~ inv_gamma(5, 5);
  sigma ~ inv_gamma(1, 1);
  marg_std ~ std_normal();
  base ~ normal(5, 10);
  response_magnitude_params ~ normal(0, 10);
  response_length_params ~ normal(0, 0.5);
  response_length_const ~ normal(0, 0.1);
  for (i in 1:n_meals) {
    meal_response_magnitudes[i] ~ inv_gamma(1, 3);
    meal_response_lengths[i] ~ inverse_gaussian(0.7, 0.3);
    //meal_response_lengths[i] ~ inverse_gaussian(3, 0.2);
  }
  meal_reporting_noise ~ normal(0, 1);
  meal_reporting_bias ~ normal(0, 0.2);

  //likelihood
  meal_timing ~ normal(meal_timing_eiv + meal_reporting_bias, meal_reporting_noise);

  // gp computations
  {
    // own local bloc for handling memory
    matrix[N, N] K = cov_exp_quad(time, marg_std, lengthscale);
    for (n in 1:N)
      K[n, n] = K[n, n] + sigma + epsilon;

    K = cholesky_decompose(K);
    glucose ~ multi_normal_cholesky(mu, K);
  }


  //pred likelihood
  for (i in n_meals+1:n_meals_pred) {
    pred_meals[i] ~ normal(pred_meals_eiv[i] + meal_reporting_bias, meal_reporting_noise);
  }
}


generated quantities {
  // same some parameter values for posterior plots
  real starch_magnitude = response_magnitude_params[1];
  real starch_lengths = response_length_params[1];
  real sugar_magnitude = response_magnitude_params[2];
  real sugar_lengths = response_length_params[2];
  real fibr_magnitude = response_magnitude_params[3];
  real fibr_lengths = response_length_params[3];
  real fat_magnitude = response_magnitude_params[4];
  real fat_lengths = response_length_params[4];
  real prot_magnitude = response_magnitude_params[5];
  real prot_lengths = response_length_params[5];

  // posterior predictive values
  vector[n_meals_pred] pred_meal_magnitudes = pred_nutrients * response_magnitude_params/n_meals;
  vector[n_meals_pred] pred_meal_lengths = pred_nutrients * response_length_params/n_meals + response_length_const;

  vector[n_pred] pred_mu = response(n_pred, n_meals_pred, to_vector(pred_times), pred_meals_eiv, pred_meal_magnitudes, pred_meal_lengths, base);
  vector[n_pred] resp = pred_mu - base;
  vector[n_pred] pred_y = draw_pred_rng(pred_times,
                     glucose,
                     mu,
                     time,
                     marg_std,
                     lengthscale,
                     epsilon,
                     pred_mu,
                     base,
                     sigma);
  vector[n_pred] baseline_gp = pred_y-resp;
  vector[n_pred] pred_gluc;
  vector[n_pred - N] test_gluc;
  vector[N] train_gluc;
  vector[N] train_y;
  vector[N] train_baseline_gp;
  train_baseline_gp = baseline_gp[:N];
  train_y = pred_y[:N];
  for (i in 1:n_pred){
    pred_gluc[i] = normal_rng(pred_y[i], sigma);
  }
  train_gluc= pred_gluc[:N];
  test_gluc = pred_gluc[N+1:];

}