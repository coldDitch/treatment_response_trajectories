functions {
   matrix impulse(real response_magnitude, real response_length, matrix delta_t) {
     return response_magnitude * exp(-0.5 * (delta_t - 3*response_length).^2 / response_length ^ 2);
   }

   vector cumulative_response(matrix y, int N, int n_meals) {
     vector[N] response = rep_vector(0, N);
     for (n in 1:n_meals) {
       response += y[n]';
     }
     return response;
   }
}

data {
  int<lower=1> N;
  real time[N];
  real lengthscale;
  real marg_std;
  real baseline;
  real response_magnitude;
  real response_length;
  int n_meals;
  real meal_reporting_noise;
  real meal_reporting_bias;
}
transformed data {
  real interval = time[N] / (n_meals+1);
  real epsilon = 1e-3;
  matrix[N, N] L;
  matrix[N, N] K = cov_exp_quad(time, marg_std, lengthscale);
  vector[N] mu = rep_vector(0, N);
  for (n in 1:N)
    K[n, n] = K[n, n] + epsilon;

  L = cholesky_decompose(K);
}

parameters {
  vector[N] eta;
}

model {
  eta ~ std_normal();
}

generated quantities {
  vector[N] glucose;
  vector[n_meals] true_timing;
  vector[n_meals] meal_timing;
  matrix[n_meals, N] time_delta;
  matrix[n_meals, N] impulses;
  vector[N] resp;
  vector[N] base_variation;
  base_variation = L * eta;
  glucose = base_variation;
  for (i in 1:n_meals) {
    true_timing[i] = normal_rng(interval * i, 1);
    time_delta[i] = to_row_vector(time) - true_timing[i];
    meal_timing[i] = normal_rng(true_timing[i] + meal_reporting_bias, meal_reporting_noise);
  }
  impulses = impulse(response_magnitude, response_length, time_delta);
  resp = cumulative_response(impulses, N, n_meals);
  glucose += resp;
  glucose += baseline;

}
