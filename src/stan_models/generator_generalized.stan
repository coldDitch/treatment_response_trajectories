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
}

data {
  int<lower=1> N;
  real time[N];
  real lengthscale;
  real marg_std;
  real baseline;
  int n_meals;
  int num_nutrients;
  vector[num_nutrients] response_magnitude_params;
  vector[num_nutrients] response_length_params;
  real meal_reporting_bias;
  real meal_reporting_noise;
  vector[num_nutrients] nutrient_mean;
  vector[num_nutrients] nutrient_std;
}
transformed data {
  vector[num_nutrients] nutrient_alpha;
  vector[num_nutrients] nutrient_beta;
  real interval = time[N] / n_meals;
  real epsilon = 1e-3;
  matrix[N, N] L;
  matrix[N, N] K = cov_exp_quad(time, marg_std, lengthscale);
  vector[N] mu = rep_vector(0, N);
  for (n in 1:N)
    K[n, n] = K[n, n] + epsilon;

  L = cholesky_decompose(K);
  if (nutrient_std[1] != 0) {
    nutrient_alpha = pow(nutrient_mean,2) ./ nutrient_std;
    nutrient_beta = pow(nutrient_mean,2) ./ nutrient_std;
  }
}

parameters {
  vector[N] eta;
}

model {
  eta ~ std_normal();
}

generated quantities {
  vector[N] glucose;
  vector[n_meals] meal_timing;
  vector[n_meals] true_timing;
  matrix[n_meals, num_nutrients] nutrients;
  matrix[n_meals, N] time_delta;
  vector[N] resp;
  vector[N] base_variation;
  vector[n_meals] meal_response_magnitudes;
  vector[n_meals] meal_response_lengths;
  base_variation = L * eta;
  glucose = base_variation;
  for (i in 1:n_meals) {
    true_timing[i] = normal_rng(interval * i, 1);
    time_delta[i] = to_row_vector(time) - true_timing[i];
    if (meal_reporting_noise == 0) {
      meal_timing[i] = true_timing[i] + meal_reporting_bias;
    }
    else{
      meal_timing[i] = normal_rng(true_timing[i] + meal_reporting_bias, meal_reporting_noise);
    }
    if (nutrient_std[1] == 0) {
      nutrients[i] = nutrient_mean';
    }
    else {
      for (j in 1:num_nutrients) {
        nutrients[i, j] = gamma_rng(nutrient_alpha[i], nutrient_beta[j]);
      }
    }
  }
  meal_response_magnitudes = nutrients * response_magnitude_params;
  meal_response_lengths = nutrients * response_length_params;
  resp = response(N, n_meals, time, true_timing, meal_response_magnitudes, meal_response_lengths, baseline)-baseline;
  glucose += resp;
  glucose += baseline;
}