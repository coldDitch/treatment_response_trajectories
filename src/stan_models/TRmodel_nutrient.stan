functions {
  matrix impulse(vector a, real response_length, matrix delta_t, matrix nutrients) {
     int n_meals = rows(delta_t);
     vector[cols(nutrients)] response_magnitude = nutrients * a;
     matrix[n_meals, cols(delta_t)] shape = exp(-0.5 * (delta_t - 3*response_length).^2 / response_length ^ 2);
     for (i in 1:n_meals){
       shape[i] = response_magnitude[i] * shape[i];
     }
     return shape;
  }

  vector response(int N, int n_meals, real[] time, vector meal_timing, matrix nutrients, vector a, real response_length, real base) {
    vector[N] mu = rep_vector(0, N) + base;
    matrix[n_meals, N] delta_t;
    matrix[n_meals, N] impulses;
    for (i in 1:n_meals) {
        delta_t[i] = to_row_vector(time)- meal_timing[i] ;
    }
    impulses = impulse(a, response_length, delta_t, nutrients);
    for (i in 1:n_meals) {
        mu += impulses[i]';
    }
    return mu;
  }
  

  vector draw_pred_rng(real[] test_x,
                       vector meal_timing,
                       matrix nutrients,
                       vector glucose,
                       vector train_mu,
                       real[] train_x,
                       real marg_std,
                       real lengthscale,
                       real epsilon,
                       vector a,
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
      mu2 = response(N2, n_meals, test_x, meal_timing, nutrients, a, response_length, base);
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
}

parameters {
  real<lower=epsilon> lengthscale;
  real<lower=0> marg_std;
  real<lower=0> base;
  vector<lower=0>[num_nutrients] a;
  real<lower=0> response_length;
}

transformed parameters {
  vector[N] mu = response(N, n_meals, time, meal_timing, nutrients, a, response_length, base);
}

model {
  vector[N] delta_t;
  matrix[N, N] L;

  // gp computations
  matrix[N, N] K = cov_exp_quad(time, marg_std, lengthscale);
  for (n in 1:N)
    K[n, n] = K[n, n] + epsilon;

  L = cholesky_decompose(K);
  
  

  //priors
  lengthscale ~ normal(5, 1);
  marg_std ~ std_normal();
  base ~ std_normal();
  response_length ~ std_normal();

  //likelihood
  glucose ~ multi_normal_cholesky(mu, L);
}

generated quantities {
  real pred_x[n_pred] = pred_times;
  vector[n_pred] pred_y;
  vector[0] empty;
  vector[n_pred] baseline = draw_pred_rng(pred_times,
                     empty,
                     pred_nutrients,
                     glucose,
                     mu,
                     time,
                     marg_std,
                     lengthscale,
                     epsilon,
                     a,
                     response_length,
                     base);
  vector[n_pred] resp = response(n_pred, n_meals_pred, pred_times, pred_meals, pred_nutrients, a, response_length, base)-base;
  pred_y = baseline + resp;
}