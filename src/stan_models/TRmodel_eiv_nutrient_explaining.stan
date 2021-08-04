
functions {
  vector impulse(real response_magnitude, real response_length, vector delta_t) {
    return response_magnitude * exp(-0.5 * (delta_t - 3*response_length).^2 / response_length ^ 2);
  }

  vector response(int N, int n_meals, real[] time, vector meal_timing, vector meal_response_magnitudes, vector meal_response_lengths, real base) {
    vector[N] mu = rep_vector(0, N) + base;
    for (i in 1:n_meals) {
        mu += impulse(meal_response_magnitudes[i], meal_response_lengths[i], to_vector(time) - meal_timing[i]);
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
                       real base) {
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
        K[n, n] = K[n,n] + epsilon;
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
  real<lower=0> base;
  vector<lower=1>[num_nutrients] response_magnitude_params;
  vector<lower=0.1>[num_nutrients] response_length_params;
  real<lower=0, upper=2> meal_reporting_noise;
  real<lower=-1, upper=1> meal_reporting_bias;
  vector[n_meals] meal_timing_eiv;
  vector[n_meals_pred-n_meals] fut_meal_timing;
}

transformed parameters {
  vector<lower=0>[n_meals] meal_response_magnitudes = nutrients * response_magnitude_params;
  vector<lower=0>[n_meals] meal_response_lengths = nutrients * response_length_params;
  vector[N] mu = response(N, n_meals, time, meal_timing_eiv, meal_response_magnitudes, meal_response_lengths, base);
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

  //abs(variance(mu) - glucose_var) ~ gamma(1, 0.01);

  // gp computations
  vector[N] delta_t;
  matrix[N, N] L;
  matrix[N, N] K = cov_exp_quad(time, marg_std, lengthscale);
  for (n in 1:N)
    K[n, n] = K[n, n] + epsilon;

  L = cholesky_decompose(K);

  //priors
  lengthscale ~ std_normal();
  marg_std ~ std_normal();
  base ~ std_normal();
  response_magnitude_params ~ normal(5, 1);
  response_length_params ~ normal(1, 1);
  meal_reporting_noise ~ std_normal();
  meal_reporting_bias ~ std_normal();

  //likelihood
  meal_timing ~ normal(meal_timing_eiv + meal_reporting_bias, meal_reporting_noise);
  glucose ~ multi_normal_cholesky(mu, L);


  //pred likelihood
  for (i in n_meals+1:n_meals_pred) {
    pred_meals[i] ~ normal(pred_meals_eiv[i] + meal_reporting_bias, meal_reporting_noise);
  }
}

generated quantities {
  real abs_vardiff = abs(variance(mu) - glucose_var);
  vector[n_meals_pred] pred_meal_magnitudes = pred_nutrients * response_magnitude_params;
  vector[n_meals_pred] pred_meal_lengths = pred_nutrients * response_length_params;
  real pred_x[n_pred] = pred_times;
  vector[n_pred] pred_mu = response(n_pred, n_meals_pred, pred_times, pred_meals_eiv, pred_meal_magnitudes, pred_meal_lengths, base);
  vector[n_pred] resp = pred_mu - base;
  vector[n_pred] pred_y = draw_pred_rng(pred_times,
                     glucose,
                     mu,
                     time,
                     marg_std,
                     lengthscale,
                     epsilon,
                     pred_mu,
                     base);
  vector[n_pred] baseline = pred_y - resp;
}
