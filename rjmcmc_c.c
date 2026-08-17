#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_21_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    uint64_t state;
    int has_normal;
    double normal_cache;
} rng_t;

static uint64_t rng_next(rng_t *rng) {
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * UINT64_C(2685821657736338717);
}

static double rng_uniform(rng_t *rng) {
    return (double)(rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

static double rng_normal(rng_t *rng, double mean, double sigma) {
    double u1, u2, radius, angle;
    if (rng->has_normal) {
        rng->has_normal = 0;
        return mean + sigma * rng->normal_cache;
    }
    do {
        u1 = rng_uniform(rng);
    } while (u1 <= 0.0);
    u2 = rng_uniform(rng);
    radius = sqrt(-2.0 * log(u1));
    angle = 2.0 * M_PI * u2;
    rng->normal_cache = radius * sin(angle);
    rng->has_normal = 1;
    return mean + sigma * radius * cos(angle);
}

static double log_normal_pdf(double x, double mean, double sigma) {
    return -0.5 * log(2.0 * M_PI * sigma * sigma) -
           ((x - mean) * (x - mean)) / (2.0 * sigma * sigma);
}

static double log_geometric_pmf(int k, double p) {
    if (k < 0 || p <= 0.0 || p > 1.0) {
        return -INFINITY;
    }
    return k * log1p(-p) + log(p);
}

static double sigmoid(double x) {
    if (x >= 0.0) {
        double z = exp(-x);
        return 1.0 / (1.0 + z);
    }
    double z = exp(x);
    return z / (1.0 + z);
}

static void theta_from_state(int k, const int64_t *taus, const double *theta,
                             int T, double *theta_t) {
    int segment = 0;
    int start = 0;
    for (segment = 0; segment <= k; ++segment) {
        int end = (segment < k) ? (int)taus[segment] : T;
        for (int t = start; t < end; ++t) {
            theta_t[t] = theta[segment];
        }
        start = end;
    }
}

static double log_likelihood(const double *deaths, const double *cases,
                             const double *delay, int delay_len, int T,
                             const double *theta_t, double *signal,
                             double *expected) {
    double log_lik = 0.0;
    for (int i = 0; i < T; ++i) {
        signal[i] = cases[i] * sigmoid(theta_t[i]);
        expected[i] = 0.0;
    }
    for (int i = 0; i < T; ++i) {
        for (int j = 0; j <= i && j < delay_len; ++j) {
            expected[i] += signal[i - j] * delay[j];
        }
    }
    for (int t = 0; t < T; ++t) {
        double mu = expected[t] > 1e-9 ? expected[t] : 1e-9;
        log_lik += deaths[t] * log(mu) - mu;
    }
    return log_lik;
}

static double theta_prior(int k, const double *theta, double mean, double sigma) {
    double result = 0.0;
    for (int j = 0; j <= k; ++j) {
        result += log_normal_pdf(theta[j], mean, sigma);
    }
    return result;
}

static int random_available_cp(rng_t *rng, const int64_t *taus, int k, int T) {
    int available = (T - 2) - k;
    int selected = (int)(rng_uniform(rng) * (double)available);
    int seen = 0;
    for (int candidate = 2; candidate < T; ++candidate) {
        int occupied = 0;
        for (int j = 0; j < k; ++j) {
            if (taus[j] == candidate) {
                occupied = 1;
                break;
            }
        }
        if (!occupied) {
            if (seen == selected) {
                return candidate;
            }
            ++seen;
        }
    }
    return T - 1;
}

static void copy_sorted_insert(const int64_t *taus, int k, int64_t tau,
                               int64_t *result) {
    int inserted = 0;
    for (int j = 0; j < k; ++j) {
        if (!inserted && tau < taus[j]) {
            result[j] = tau;
            inserted = 1;
        }
        result[j + inserted] = taus[j];
    }
    if (!inserted) {
        result[k] = tau;
    }
}

static void copy_sorted(const int64_t *source, int k, int64_t *result) {
    for (int i = 0; i < k; ++i) {
        result[i] = source[i];
    }
    for (int i = 1; i < k; ++i) {
        int64_t value = result[i];
        int j = i - 1;
        while (j >= 0 && result[j] > value) {
            result[j + 1] = result[j];
            --j;
        }
        result[j + 1] = value;
    }
}

static void move_probabilities(int k, int k_max, double *birth, double *death,
                               double *move, double *update) {
    if (k_max == 0) {
        *birth = 0.0;
        *death = 0.0;
        *move = 0.0;
        *update = 1.0;
    } else if (k == 0) {
        *birth = 0.25;
        *death = 0.0;
        *move = 0.0;
        *update = 0.75;
    } else if (k == k_max) {
        *birth = 0.0;
        *death = 0.25;
        *move = 0.25;
        *update = 0.50;
    } else {
        *birth = 0.25;
        *death = 0.25;
        *move = 0.25;
        *update = 0.25;
    }
}

static int relocation_support(const int64_t *taus, int k, int idx, int T,
                              int move_window, int *lower, int *upper) {
    int current = (int)taus[idx];
    *lower = current - move_window;
    *upper = current + move_window;
    if (*lower < 2) *lower = 2;
    if (*upper > T - 1) *upper = T - 1;
    if (idx > 0 && *lower < taus[idx - 1] + 1) {
        *lower = (int)taus[idx - 1] + 1;
    }
    if (idx < k - 1 && *upper > taus[idx + 1] - 1) {
        *upper = (int)taus[idx + 1] - 1;
    }
    return *upper > *lower ? *upper - *lower : 0;
}

static double relocation_mixture_probability(
        const int64_t *taus, int k, int idx, int T, int move_window,
        int candidate, double global_move_prob) {
    int local_lower, local_upper, global_lower, global_upper;
    int local_size = relocation_support(
        taus, k, idx, T, move_window, &local_lower, &local_upper);
    int global_size = relocation_support(
        taus, k, idx, T, T, &global_lower, &global_upper);
    double probability = 0.0;
    if (global_size > 0 && candidate >= global_lower && candidate <= global_upper) {
        probability += global_move_prob / (double)global_size;
    }
    if (local_size > 0 && candidate >= local_lower && candidate <= local_upper) {
        probability += (1.0 - global_move_prob) / (double)local_size;
    }
    return probability;
}

static PyObject *rjmcmc_sample(PyObject *self, PyObject *args) {
    PyObject *deaths_obj, *cases_obj, *delay_obj;
    PyObject *initial_taus_obj, *initial_theta_obj;
    PyArrayObject *deaths_array = NULL, *cases_array = NULL, *delay_array = NULL;
    PyArrayObject *initial_taus_array = NULL, *initial_theta_array = NULL;
    double p_geom, theta_mu, theta_sigma, u_sigma, theta_prop_sigma;
    double global_move_prob;
    int move_window, mcmc_iter, burn_in, k_max, initial_k, return_diagnostics;
    unsigned long long seed;
    int T, delay_len;
    npy_intp dims[1], tau_dims[2], theta_dims[2];
    PyArrayObject *k_samples = NULL, *taus_samples = NULL, *theta_samples = NULL;
    double *theta_t = NULL, *signal = NULL, *expected = NULL;
    int64_t *taus = NULL, *taus_new = NULL;
    double *theta = NULL, *theta_new = NULL;
    rng_t rng;
    int64_t proposed[4] = {0, 0, 0, 0};
    int64_t accepted[4] = {0, 0, 0, 0};

    if (!PyArg_ParseTuple(args, "OOOdddddidiiiKiOOi", &deaths_obj, &cases_obj,
                          &delay_obj, &p_geom, &theta_mu, &theta_sigma,
                          &u_sigma, &theta_prop_sigma, &move_window,
                          &global_move_prob,
                          &mcmc_iter, &burn_in, &k_max, &seed, &initial_k,
                          &initial_taus_obj, &initial_theta_obj,
                          &return_diagnostics)) {
        return NULL;
    }

    deaths_array = (PyArrayObject *)PyArray_FROM_OTF(
        deaths_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    cases_array = (PyArrayObject *)PyArray_FROM_OTF(
        cases_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    delay_array = (PyArrayObject *)PyArray_FROM_OTF(
        delay_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    initial_taus_array = (PyArrayObject *)PyArray_FROM_OTF(
        initial_taus_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    initial_theta_array = (PyArrayObject *)PyArray_FROM_OTF(
        initial_theta_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (deaths_array == NULL || cases_array == NULL || delay_array == NULL ||
        initial_taus_array == NULL || initial_theta_array == NULL) {
        goto error;
    }
    if (PyArray_NDIM(deaths_array) != 1 || PyArray_NDIM(cases_array) != 1 ||
        PyArray_NDIM(delay_array) != 1 || PyArray_NDIM(initial_taus_array) != 1 ||
        PyArray_NDIM(initial_theta_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "deaths, cases, and delay_pmf must be 1-D");
        goto error;
    }
    T = (int)PyArray_DIM(deaths_array, 0);
    if (PyArray_DIM(cases_array, 0) != T || T <= 0 || mcmc_iter < 0 ||
        burn_in < 0 || k_max < 0 || k_max >= T || theta_sigma <= 0.0 ||
        u_sigma <= 0.0 || theta_prop_sigma <= 0.0 || move_window < 0 ||
        global_move_prob < 0.0 || global_move_prob > 1.0 ||
        initial_k < -1 || initial_k > k_max) {
        PyErr_SetString(PyExc_ValueError, "invalid sampler dimensions or parameters");
        goto error;
    }
    delay_len = (int)PyArray_DIM(delay_array, 0);
    if (delay_len <= 0) {
        PyErr_SetString(PyExc_ValueError, "delay_pmf must not be empty");
        goto error;
    }
    if (initial_k >= 0 &&
        (PyArray_DIM(initial_taus_array, 0) != initial_k ||
         PyArray_DIM(initial_theta_array, 0) != initial_k + 1)) {
        PyErr_SetString(PyExc_ValueError, "initial state arrays have invalid lengths");
        goto error;
    }

    dims[0] = (npy_intp)mcmc_iter;
    tau_dims[0] = (npy_intp)mcmc_iter;
    tau_dims[1] = (npy_intp)k_max;
    theta_dims[0] = (npy_intp)mcmc_iter;
    theta_dims[1] = (npy_intp)(k_max + 1);
    k_samples = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_INT64, 0);
    taus_samples = (PyArrayObject *)PyArray_EMPTY(2, tau_dims, NPY_INT64, 0);
    theta_samples = (PyArrayObject *)PyArray_EMPTY(2, theta_dims, NPY_DOUBLE, 0);
    if (k_samples == NULL || taus_samples == NULL || theta_samples == NULL) {
        goto error;
    }
    for (npy_intp i = 0; i < (npy_intp)mcmc_iter * k_max; ++i) {
        ((npy_int64 *)PyArray_DATA(taus_samples))[i] = -1;
    }
    for (npy_intp i = 0; i < (npy_intp)mcmc_iter * (k_max + 1); ++i) {
        ((double *)PyArray_DATA(theta_samples))[i] = NAN;
    }

    theta_t = (double *)malloc((size_t)T * sizeof(double));
    signal = (double *)malloc((size_t)T * sizeof(double));
    expected = (double *)malloc((size_t)T * sizeof(double));
    taus = (int64_t *)calloc((size_t)(k_max > 0 ? k_max : 1), sizeof(int64_t));
    taus_new = (int64_t *)calloc((size_t)(k_max > 0 ? k_max : 1), sizeof(int64_t));
    theta = (double *)malloc((size_t)(k_max + 1) * sizeof(double));
    theta_new = (double *)malloc((size_t)(k_max + 1) * sizeof(double));
    if (theta_t == NULL || signal == NULL || expected == NULL || taus == NULL || taus_new == NULL ||
        theta == NULL || theta_new == NULL) {
        PyErr_NoMemory();
        goto error;
    }

    rng.state = (uint64_t)seed;
    if (rng.state == 0) {
        rng.state = UINT64_C(88172645463393265);
    }
    rng.has_normal = 0;
    rng.normal_cache = 0.0;
    int k = initial_k >= 0 ? initial_k : 0;
    if (initial_k >= 0) {
        if (initial_k > 0) {
            memcpy(taus, PyArray_DATA(initial_taus_array),
                   (size_t)initial_k * sizeof(int64_t));
        }
        memcpy(theta, PyArray_DATA(initial_theta_array),
               (size_t)(initial_k + 1) * sizeof(double));
    } else {
        theta[0] = rng_normal(&rng, theta_mu, theta_sigma);
    }

    Py_BEGIN_ALLOW_THREADS
    for (int iter_idx = 0; iter_idx < mcmc_iter + burn_in; ++iter_idx) {
        theta_from_state(k, taus, theta, T, theta_t);
        double log_lik_current = log_likelihood(
            (const double *)PyArray_DATA(deaths_array),
            (const double *)PyArray_DATA(cases_array),
            (const double *)PyArray_DATA(delay_array), delay_len, T,
            theta_t, signal, expected);
        double log_prior_k_current = log_geometric_pmf(k, p_geom);
        double log_prior_theta_current = theta_prior(k, theta, theta_mu, theta_sigma);
        double u_move = rng_uniform(&rng);
        double p_birth, p_death, p_move, p_update;
        move_probabilities(k, k_max, &p_birth, &p_death, &p_move, &p_update);
        int move_type;
        if (u_move < p_birth) {
            move_type = 0; /* birth */
        } else if (u_move < p_birth + p_death) {
            move_type = 1; /* death */
        } else if (u_move < p_birth + p_death + p_move) {
            move_type = 2; /* move */
        } else {
            move_type = 3; /* update */
        }
        proposed[move_type] += (move_type == 3) ? (k + 1) : 1;

        if (move_type == 0) {
            int tau_star = random_available_cp(&rng, taus, k, T);
            int seg_idx = 0;
            while (seg_idx < k && taus[seg_idx] < tau_star) {
                ++seg_idx;
            }
            double u_aux = rng_normal(&rng, 0.0, u_sigma);
            double theta1_star = theta[seg_idx] - u_aux;
            double theta2_star = theta[seg_idx] + u_aux;
            copy_sorted_insert(taus, k, (int64_t)tau_star, taus_new);
            for (int j = 0; j < seg_idx; ++j) {
                theta_new[j] = theta[j];
            }
            theta_new[seg_idx] = theta1_star;
            theta_new[seg_idx + 1] = theta2_star;
            for (int j = seg_idx + 1; j <= k; ++j) {
                theta_new[j + 1] = theta[j];
            }
            theta_from_state(k + 1, taus_new, theta_new, T, theta_t);
            double log_lik_new = log_likelihood(
                (const double *)PyArray_DATA(deaths_array),
                (const double *)PyArray_DATA(cases_array),
                (const double *)PyArray_DATA(delay_array), delay_len, T,
                theta_t, signal, expected);
            double log_prior_k_new = log_geometric_pmf(k + 1, p_geom);
            double log_prior_theta_new = theta_prior(k + 1, theta_new, theta_mu, theta_sigma);
            double p_birth_new, p_death_new, ignored_move, ignored_update;
            move_probabilities(k + 1, k_max, &p_birth_new, &p_death_new,
                               &ignored_move, &ignored_update);
            /* Uniform location-prior and location-selection counts cancel. */
            double log_alpha = (log_lik_new - log_lik_current) +
                (log_prior_k_new - log_prior_k_current) +
                (log_prior_theta_new - log_prior_theta_current) -
                log_normal_pdf(u_aux, 0.0, u_sigma) + log(2.0) +
                log(p_death_new / p_birth);
            if (log(rng_uniform(&rng)) < log_alpha) {
                ++k;
                memcpy(taus, taus_new, (size_t)k * sizeof(int64_t));
                memcpy(theta, theta_new, (size_t)(k + 1) * sizeof(double));
                accepted[move_type] += 1;
            }
        } else if (move_type == 1 && k > 0) {
            int idx = (int)(rng_uniform(&rng) * (double)k);
            double theta1 = theta[idx];
            double theta2 = theta[idx + 1];
            double theta_j_star = (theta1 + theta2) / 2.0;
            double u_aux = (theta2 - theta1) / 2.0;
            for (int j = 0; j < idx; ++j) {
                taus_new[j] = taus[j];
                theta_new[j] = theta[j];
            }
            for (int j = idx; j < k - 1; ++j) {
                taus_new[j] = taus[j + 1];
            }
            theta_new[idx] = theta_j_star;
            for (int j = idx + 1; j < k; ++j) {
                theta_new[j] = theta[j + 1];
            }
            theta_from_state(k - 1, taus_new, theta_new, T, theta_t);
            double log_lik_new = log_likelihood(
                (const double *)PyArray_DATA(deaths_array),
                (const double *)PyArray_DATA(cases_array),
                (const double *)PyArray_DATA(delay_array), delay_len, T,
                theta_t, signal, expected);
            double log_prior_k_new = log_geometric_pmf(k - 1, p_geom);
            double log_prior_theta_new = theta_prior(k - 1, theta_new, theta_mu, theta_sigma);
            double p_birth_new, p_death_new, ignored_move, ignored_update;
            move_probabilities(k - 1, k_max, &p_birth_new, &p_death_new,
                               &ignored_move, &ignored_update);
            double log_alpha = (log_lik_new - log_lik_current) +
                (log_prior_k_new - log_prior_k_current) +
                (log_prior_theta_new - log_prior_theta_current) +
                log_normal_pdf(u_aux, 0.0, u_sigma) - log(2.0) +
                log(p_birth_new / p_death);
            if (log(rng_uniform(&rng)) < log_alpha) {
                --k;
                memcpy(taus, taus_new, (size_t)k * sizeof(int64_t));
                memcpy(theta, theta_new, (size_t)(k + 1) * sizeof(double));
                accepted[move_type] += 1;
            }
        } else if (move_type == 2 && k > 0) {
            int idx = (int)(rng_uniform(&rng) * (double)k);
            int64_t tau_current = taus[idx];
            int active_window = rng_uniform(&rng) < global_move_prob ? T : move_window;
            int lower_prop, upper_prop;
            int support_size = relocation_support(taus, k, idx, T, active_window,
                                                  &lower_prop, &upper_prop);
            if (support_size > 0) {
                int proposal_idx = (int)(rng_uniform(&rng) * (double)support_size);
                int tau_new = lower_prop + proposal_idx;
                if (tau_new >= taus[idx]) ++tau_new;
                copy_sorted(taus, k, taus_new);
                taus_new[idx] = tau_new;
                copy_sorted(taus_new, k, taus_new);
                theta_from_state(k, taus_new, theta, T, theta_t);
                double log_lik_new = log_likelihood(
                    (const double *)PyArray_DATA(deaths_array),
                    (const double *)PyArray_DATA(cases_array),
                    (const double *)PyArray_DATA(delay_array), delay_len, T,
                    theta_t, signal, expected);
                double q_forward = relocation_mixture_probability(
                    taus, k, idx, T, move_window, tau_new, global_move_prob);
                double q_reverse = relocation_mixture_probability(
                    taus_new, k, idx, T, move_window, (int)tau_current,
                    global_move_prob);
                double log_alpha = log_lik_new - log_lik_current +
                    log(q_reverse / q_forward);
                if (log(rng_uniform(&rng)) < log_alpha) {
                    memcpy(taus, taus_new, (size_t)k * sizeof(int64_t));
                    accepted[move_type] += 1;
                }
            }
        } else if (move_type == 3) {
            for (int j = 0; j <= k; ++j) {
                double theta_proposal = rng_normal(&rng, theta[j], theta_prop_sigma);
                memcpy(theta_new, theta, (size_t)(k + 1) * sizeof(double));
                theta_new[j] = theta_proposal;
                theta_from_state(k, taus, theta_new, T, theta_t);
                double log_lik_new = log_likelihood(
                    (const double *)PyArray_DATA(deaths_array),
                    (const double *)PyArray_DATA(cases_array),
                    (const double *)PyArray_DATA(delay_array), delay_len, T,
                    theta_t, signal, expected);
                double log_alpha = (log_lik_new - log_lik_current) +
                    log_normal_pdf(theta_proposal, theta_mu, theta_sigma) -
                    log_normal_pdf(theta[j], theta_mu, theta_sigma);
                if (log(rng_uniform(&rng)) < log_alpha) {
                    theta[j] = theta_proposal;
                    log_lik_current = log_lik_new;
                    accepted[move_type] += 1;
                }
            }
        }

        if (iter_idx >= burn_in) {
            int sample_idx = iter_idx - burn_in;
            ((npy_int64 *)PyArray_DATA(k_samples))[sample_idx] = (npy_int64)k;
            for (int j = 0; j < k; ++j) {
                ((npy_int64 *)PyArray_DATA(taus_samples))[sample_idx * k_max + j] = (npy_int64)taus[j];
            }
            for (int j = 0; j <= k; ++j) {
                ((double *)PyArray_DATA(theta_samples))[sample_idx * (k_max + 1) + j] = theta[j];
            }
        }
    }
    Py_END_ALLOW_THREADS

    free(theta_t);
    free(signal);
    free(expected);
    free(taus);
    free(taus_new);
    free(theta);
    free(theta_new);
    Py_DECREF(deaths_array);
    Py_DECREF(cases_array);
    Py_DECREF(delay_array);
    Py_DECREF(initial_taus_array);
    Py_DECREF(initial_theta_array);
    if (return_diagnostics) {
        PyObject *diagnostics = Py_BuildValue(
            "{s:[L,L,L,L],s:[L,L,L,L],s:(s,s,s,s)}",
            "proposed", (long long)proposed[0], (long long)proposed[1],
            (long long)proposed[2], (long long)proposed[3],
            "accepted", (long long)accepted[0], (long long)accepted[1],
            (long long)accepted[2], (long long)accepted[3],
            "move_names", "birth", "death", "move", "update");
        if (diagnostics == NULL) {
            Py_DECREF(k_samples);
            Py_DECREF(taus_samples);
            Py_DECREF(theta_samples);
            return NULL;
        }
        return Py_BuildValue("NNNN", k_samples, taus_samples, theta_samples,
                             diagnostics);
    }
    return Py_BuildValue("NNN", k_samples, taus_samples, theta_samples);

error:
    free(theta_t);
    free(signal);
    free(expected);
    free(taus);
    free(taus_new);
    free(theta);
    free(theta_new);
    Py_XDECREF(k_samples);
    Py_XDECREF(taus_samples);
    Py_XDECREF(theta_samples);
    Py_XDECREF(deaths_array);
    Py_XDECREF(cases_array);
    Py_XDECREF(delay_array);
    Py_XDECREF(initial_taus_array);
    Py_XDECREF(initial_theta_array);
    return NULL;
}

static PyMethodDef methods[] = {
    {"sample", rjmcmc_sample, METH_VARARGS,
     "Run the C RJMCMC kernel and return (k, taus, theta) samples."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_rjmcmc_c",
    "Optional C backend for the RJMCMC-CFR sampler.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__rjmcmc_c(void) {
    import_array();
    return PyModule_Create(&module);
}
