functions {
   matrix impulse(real a, real b, matrix delta_t) {
     return a * exp(-0.5 * (delta_t - 3*b).^2 / b ^ 2);
   }

   vector cumulative_response(matrix y, int N, int meal_count) {
     vector[N] response = rep_vector(0, N);
     for (n in 1:meal_count) {
       response += y[n]';
     }
     return response;
   }

}

data {
  int<lower=1> N;
  real t_meas[N];
  real lengthscale;
  real marg_std;
  real baseline;
  real a;
  real b;
  int meal_count;
}
transformed data {
  real interval = t_meas[N] / meal_count;
  real epsilon = 1e-3;
  matrix[N, N] L;
  matrix[N, N] K = cov_exp_quad(t_meas, marg_std, lengthscale);
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
  vector[N] y;
  vector[meal_count] meals;
  matrix[meal_count, N] time_delta;
  matrix[meal_count, N] impulses;
  vector[N] resp;
  vector[N] base_variation;
  base_variation = L * eta;
  y = base_variation;
  for (i in 1:meal_count) {
    meals[i] = normal_rng(interval * i, 1);
    time_delta[i] = to_row_vector(t_meas) - meals[i];
  }
  impulses = impulse(a, b, time_delta);
  resp = cumulative_response(impulses, N, meal_count);
  y += resp;
  y += baseline;

}