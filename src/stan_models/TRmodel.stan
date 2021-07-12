functions {
  vector impulse(real a, real b, vector delta_t) {
    return a * exp(-0.5 * (delta_t - 3*b).^2 / b ^ 2);
  }

  vector response(int N, int n_meals, real[] time, vector meal_timing, real a, real b, real base) {
    vector[N] mu = rep_vector(0, N) + base;
    for (i in 1:n_meals) {
        mu += impulse(a, b, to_vector(time) - meal_timing[i]);
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
                       real a,
                       real b, 
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
      mu2 = response(N2, n_meals, test_x, meal_timing, a, b, base);
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
  int pred_n;
  real pred_times[pred_n];
  int pred_mn;
  vector[pred_mn] pred_meals;
}
transformed data {
  real epsilon = 1e-3;
}

parameters {
  real<lower=epsilon> lengthscale;
  real<lower=0> marg_std;
  real<lower=0> base;
  real<lower=0> a;
  real<lower=0> b;
}

transformed parameters {
  vector[N] mu = response(N, n_meals, time, meal_timing, a, b, base);
}

model {
  // gp computations

  vector[N] delta_t;
  matrix[N, N] L;
  matrix[N, N] K = cov_exp_quad(time, marg_std, lengthscale);
  for (n in 1:N)
    K[n, n] = K[n, n] + epsilon;

  L = cholesky_decompose(K);
  
  

  //priors
  lengthscale ~ normal(5, 1);
  marg_std ~ std_normal();
  base ~ std_normal();
  b ~ std_normal();

  //likelihood
  glucose ~ multi_normal_cholesky(mu, L);
}

generated quantities {
  real pred_x[pred_n] = pred_times;
  vector[pred_n] pred_y = draw_pred_rng(pred_times,
                     pred_meals,
                     glucose,
                     mu,
                     time,
                     marg_std,
                     lengthscale,
                     epsilon,
                     a,
                     b,
                     base);
  vector[0] empty;
  vector[pred_n] baseline = draw_pred_rng(pred_times,
                     empty,
                     glucose,
                     mu,
                     time,
                     marg_std,
                     lengthscale,
                     epsilon,
                     a,
                     b,
                     base);
  vector[pred_n] resp = response(pred_n, pred_mn, pred_times, pred_meals, a, b, base)-base;
}