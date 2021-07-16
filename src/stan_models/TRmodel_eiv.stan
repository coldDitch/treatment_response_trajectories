functions {
  vector impulse(real response_magnitude, real response_length, vector delta_t) {
    return response_magnitude * exp(-0.5 * (delta_t - 3*response_length).^2 / response_length ^ 2);
  }

  vector response(int N, int n_meals, real[] time, vector meal_timing, real response_magnitude, real response_length, real base) {
    vector[N] mu = rep_vector(0, N) + base;
    for (i in 1:n_meals) {
        mu += impulse(response_magnitude, response_length, to_vector(time) - meal_timing[i]);
    }
    return mu;
  }

  vector draw_pred_rng(real[] test_x,
                       vector meal_timing,
                       vector glucose,
                       vector train_mu,
                       real[] train_x,
                       real marg_std,
                       real lengthscale,
                       real epsilon,
                       real response_magnitude,
                       real response_length, 
                       real base) {
    int N1 = rows(glucose);
    int N2 = size(test_x);
    int n_meals = size(meal_timing);
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
      vector[N2] mu2;
      mu2 = response(N2, n_meals, test_x, meal_timing, response_magnitude, response_length, base);
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
}
transformed data {
  real epsilon = 1e-3;
}

parameters {
  real<lower=epsilon> lengthscale;
  real<lower=0> marg_std;
  real<lower=0> base;
  real<lower=0> response_magnitude;
  real<lower=0> response_length;
  real<lower=0> meal_reporting_noise;
  real meal_reporting_bias;
  vector[n_meals] meal_timing_eiv;
}


model {
  // gp computations
  vector[N] mu;
  vector[N] delta_t;
  matrix[N, N] L;
  matrix[N, N] K = cov_exp_quad(time, marg_std, lengthscale);
  for (n in 1:N)
    K[n, n] = K[n, n] + epsilon;

  L = cholesky_decompose(K);

  //priors
  lengthscale ~ normal(5, 0.5);
  marg_std ~ normal(0, 0.5);
  base ~ std_normal();
  response_magnitude ~ uniform(1, 10);
  response_length ~ uniform(0.1, 0.5);
  meal_reporting_noise ~ normal(0, 0.1);
  meal_reporting_bias ~ normal(0, 0.1);

  //likelihood
  meal_timing ~ normal(meal_timing_eiv + meal_reporting_bias, meal_reporting_noise);
  mu = response(N, n_meals, time, meal_timing_eiv, response_magnitude, response_length, base);
  glucose ~ multi_normal_cholesky(mu, L);
}

generated quantities {
 // vector[n_meals] meal_timing_eiv_rng = to_vector(normal_rng(meal_timing - meal_reporting_bias, meal_reporting_noise));
  vector[N] mu = response(N, n_meals, time, meal_timing_eiv, response_magnitude, response_length, base);
  real pred_x[n_pred] = pred_times;
  vector[n_meals_pred] pred_meals_eiv = to_vector(normal_rng(pred_meals + meal_reporting_bias, meal_reporting_noise));
  vector[n_pred] pred_y;
  vector[0] empty;
  vector[n_pred] baseline = draw_pred_rng(pred_times,
                     empty,
                     glucose,
                     mu,
                     time,
                     marg_std,
                     lengthscale,
                     epsilon,
                     response_magnitude,
                     response_length,
                     base);
  vector[n_pred] resp = response(n_pred, n_meals_pred, pred_times, to_vector(pred_meals_eiv), response_magnitude, response_length, base)-base;
  pred_y = baseline + resp;
}