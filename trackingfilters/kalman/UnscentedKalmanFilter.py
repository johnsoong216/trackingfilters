import numpy as np
import scipy.linalg as sla
from trackingfilters.kalman.BaseKalman import BaseKalmanFilter
from trackingfilters.exceptions import UnscentedKalmanException


class UnscentedKalmanFilter(BaseKalmanFilter):

    def __init__(self, init_state_vec, init_state_cov_mat, process_noise_mat,
                 measure_noise_mat, transition_func, observation_func, alpha, beta, kappa):

        super(UnscentedKalmanFilter, self).__init__(init_state_vec=init_state_vec,
                                                    init_state_cov_mat=init_state_cov_mat,
                                                    process_noise_mat=process_noise_mat,
                                                    measure_noise_mat=measure_noise_mat)

        UnscentedKalmanFilter.dimension_check(
            init_state_vec.shape,
            init_state_cov_mat.shape,
            BaseKalmanFilter.mat_shape(process_noise_mat),
            BaseKalmanFilter.mat_shape(measure_noise_mat)
        )

        self.func_check(transition_func, init_state_vec, init_state_vec.shape)

        self.transition_func = transition_func
        self.observation_func = observation_func

        self.scaling = np.square(alpha) * (init_state_vec.shape[0] + kappa) - init_state_vec.shape[0]
        self.sigma_dim = 1 + 2 * init_state_vec.shape[0]
        self.state_weight, self.cov_weight = UnscentedKalmanFilter.calc_state_cov_weight(init_state_vec.shape[0], self.scaling, alpha, beta)

    def extrapolate(self, cur_state_vec, cur_state_cov_mat, transition_func, observation_func, measure_noise_mat,
                    process_noise_mat):

        cur_sigma_mat = UnscentedKalmanFilter.calc_sigma_mat(cur_state_vec, cur_state_cov_mat,
                                                             self.scaling, self.sigma_dim)

        extrapolate_state_vec, extrapolate_state_mat = UnscentedKalmanFilter.state_extrapolation(cur_sigma_mat,
                                                                                                 self.transition_func,
                                                                                                 self.state_weight)

        extrapolate_cov_mat = UnscentedKalmanFilter.cov_extrapolation(self.cov_weight, extrapolate_state_mat,
                                                                      extrapolate_state_vec, self.process_noise_mat)

        extrapolate_measure_vec, extrapolate_measure_mat = UnscentedKalmanFilter.measure_extrapolation(
            extrapolate_state_mat, observation_func, self.state_weight)

        measure_cov_mat, state_measure_cov_mat = UnscentedKalmanFilter.calc_unscented_cov_mat(
            self.cov_weight, extrapolate_state_mat, extrapolate_measure_mat,
                              extrapolate_state_vec, extrapolate_measure_vec, self.measure_noise_mat)

        kalman_gain_mat = UnscentedKalmanFilter.calc_kalman_gain(measure_cov_mat, state_measure_cov_mat)
        return extrapolate_state_vec, extrapolate_measure_vec, extrapolate_cov_mat, kalman_gain_mat, measure_cov_mat

    def predict(self, extrapolate_state_vec, extrapolate_measure_vec, extrapolate_cov_mat, measure_vec,
                kalman_gain_mat, measure_cov_mat):

        next_state_vec = UnscentedKalmanFilter.state_update(extrapolate_state_vec, extrapolate_measure_vec, measure_vec, kalman_gain_mat)
        next_state_cov_mat = UnscentedKalmanFilter.cov_update(extrapolate_cov_mat, measure_cov_mat, kalman_gain_mat)
        return next_state_vec, next_state_cov_mat

    def vec_update(self, measure_vec):

        self.measure_check(measure_vec)

        for measure in measure_vec:
            if isinstance(measure, np.ndarray):
                measure = measure.reshape(-1, 1)
            self.step_update(measure)

    def step_update(self, measure_vec, cur_state_vec=None, cur_state_cov_mat=None, transition_func=None,
                    observation_func=None, measure_noise_mat=None,
                    process_noise_mat=None):

        self.update_external_params(cur_state_vec, cur_state_cov_mat, transition_func, observation_func, measure_noise_mat,
                                    process_noise_mat)

        self.func_check(self.observation_func, measure_vec, self.cur_state_vec)

        try:
            extrapolate_state_vec, extrapolate_measure_vec, extrapolate_cov_mat, kalman_gain_mat, measure_cov_mat = self.extrapolate(
                self.cur_state_vec,
                self.cur_state_cov_mat,
                self.transition_func if type(self.transition_func) != list else self.transition_func[self.state_count],
                self.observation_func if type(self.observation_func) != list else self.observation_func[self.state_count],
                self.measure_noise_mat if self.measure_noise_mat.ndim == 2 else self.measure_noise_mat[self.state_count],
                self.process_noise_mat if self.process_noise_mat.ndim == 2 else self.process_noise_mat[self.state_count]
            )

            next_state_vec, next_state_cov_mat = self.predict(
                extrapolate_state_vec,
                extrapolate_measure_vec,
                extrapolate_cov_mat,
                measure_vec,
                kalman_gain_mat,
                measure_cov_mat
            )

        except IndexError:
            raise UnscentedKalmanException("List of Matrices passed in is less than the number of observations.")

        self.record_update(kalman_gain_mat, next_state_cov_mat, next_state_vec)
        return next_state_vec, next_state_cov_mat, kalman_gain_mat

    @staticmethod
    def dimension_check(init_state_vec_s, init_state_cov_mat_s, process_noise_mat_s, measure_noise_mat_s):

        if not (init_state_vec_s[1] == 1):
            raise UnscentedKalmanException("Init State Vector is not in column vector format")

        if init_state_cov_mat_s != process_noise_mat_s:
            raise UnscentedKalmanException(
                "Init State Covariance Matrix does not have the same shape as the Process Noise Matrix")

        if init_state_vec_s[0] != init_state_cov_mat_s[0]:
            raise UnscentedKalmanException("Init State Covariance Matrix and Init State Vector do not have consistent shape")

    @staticmethod
    def func_check(func, input_vec, output_vec_dim):
        if func(input_vec).shape != output_vec_dim:
            raise UnscentedKalmanException("Function Input/Output Size has a mismatch")

    def update_external_params(self, cur_state_vec, cur_state_cov_mat, transition_func, observation_func, measure_noise_mat,
                               process_noise_mat):
        if any([x.dim > 2 for x in
                [cur_state_vec, cur_state_cov_mat, measure_noise_mat, process_noise_mat] if
                x is not None]):
            raise UnscentedKalmanException(
                "Filter does not support inputting a list of matrices/vecs when doing step-wise update")

        if cur_state_vec is not None:
            self.cur_state_vec = cur_state_vec
        if cur_state_cov_mat is not None:
            self.cur_state_cov_mat = cur_state_cov_mat
        if transition_func is not None:
            self.transition_func = transition_func
        if observation_func is not None:
            self.observation_func = observation_func
        if measure_noise_mat is not None:
            self.measure_noise_mat = measure_noise_mat
        if process_noise_mat is not None:
            self.process_noise_mat = process_noise_mat

    @staticmethod
    def calc_state_cov_weight(state_vec_dim, scaling_param, alpha, beta):
        state_weight = np.zeros((1 + state_vec_dim * 2))
        cov_weight = np.zeros((1 + state_vec_dim * 2))

        state_weight[0] = scaling_param/(state_vec_dim + scaling_param)
        cov_weight[0] = scaling_param/(state_vec_dim + scaling_param) + (1 - alpha **2 + beta)

        state_weight[1:] = 1/(2 * (state_vec_dim + scaling_param))
        cov_weight[1:] = 1/(2 * (state_vec_dim + scaling_param))
        return state_weight, cov_weight

    @staticmethod
    def calc_sigma_mat(cur_state_vec, cur_cov_mat, scaling_param, sigma_dim):
        cur_state_dim = cur_state_vec.shape[0]
        sigma_mat = np.float64(np.repeat(cur_state_vec, sigma_dim, axis=1))
        scale_mat = sla.sqrtm(cur_cov_mat * (cur_state_dim + scaling_param))

        sigma_mat[:, 1:cur_state_dim + 1] += scale_mat
        sigma_mat[:, cur_state_dim + 1:] -= scale_mat
        return sigma_mat

    @staticmethod
    def state_extrapolation(sigma_mat, transition_func, state_weight):

        extrapolate_state_mat = np.zeros(shape=sigma_mat.shape)
        extrapolate_state_vec = np.zeros(shape=(sigma_mat.shape[0], 1))

        for individual_vec in range(sigma_mat.shape[1]):
            extrapolate_state_mat[:, individual_vec] = transition_func(sigma_mat[:, individual_vec])

        for individual_vec in range(extrapolate_state_mat.shape[1]):
            extrapolate_state_vec += (state_weight[individual_vec] * extrapolate_state_mat[:, individual_vec]).reshape(-1, 1)

        return extrapolate_state_vec, extrapolate_state_mat

    @staticmethod
    def cov_extrapolation(cov_weight, extrapolate_state_mat, extrapolate_state_vec, process_noise_mat):
        extrapolate_cov_mat = np.zeros(shape=(len(cov_weight), len(cov_weight)))
        for idx, weight in enumerate(cov_weight):
            temp_diff = extrapolate_state_mat[:, idx].reshape(-1, 1) - extrapolate_state_vec
            extrapolate_cov_mat += weight * (temp_diff @ temp_diff.T)
        return extrapolate_cov_mat + process_noise_mat

    @staticmethod
    def measure_extrapolation(extrapolate_state_mat, observation_func, state_weight):

        measure_dim = observation_func(extrapolate_state_mat[:, 0]).shape[0]

        extrapolate_measure_mat = np.zeros(shape=(measure_dim, extrapolate_state_mat.shape[1]))
        extrapolate_measure_vec = np.zeros(shape=(measure_dim, 1))

        for individual_vec in range(extrapolate_state_mat.shape[1]):
            extrapolate_measure_mat[:, individual_vec] = observation_func(extrapolate_state_mat[:, individual_vec])

        for individual_vec in range(extrapolate_measure_mat.shape[1]):
            extrapolate_measure_vec += (
                        state_weight[individual_vec] * extrapolate_measure_mat[:, individual_vec]).reshape(-1, 1)

        return extrapolate_measure_vec, extrapolate_measure_mat
    
    @staticmethod
    def calc_unscented_cov_mat(cov_weight, extrapolate_state_mat, extrapolate_measure_mat, extrapolate_state_vec,
                               extrapolate_measure_vec, measure_noise_mat):

        measure_cov_mat = np.zeros(shape=(len(cov_weight), len(cov_weight)))  # Measure Variance
        state_measure_cov_mat = np.zeros(shape=(len(cov_weight), len(cov_weight)))  # Measure State Covariance

        for idx, weight in enumerate(cov_weight):
            state_diff = extrapolate_state_mat[:, idx].reshape(-1, 1) - extrapolate_state_vec
            measure_diff = extrapolate_measure_mat[:, idx].reshape(-1, 1) - extrapolate_measure_vec

            measure_cov_mat += weight * (measure_diff @ measure_diff.T)
            state_measure_cov_mat += weight * (state_diff @ measure_diff.T)

        measure_cov_mat += measure_noise_mat
        return measure_cov_mat, state_measure_cov_mat

    @staticmethod
    def calc_kalman_gain(measure_cov_mat, state_measure_cov_mat):
        return state_measure_cov_mat @ sla.inv(measure_cov_mat)

    @staticmethod
    def state_update(extrapolate_state_vec, extrapolate_measure_vec, measure_vec, kalman_gain_mat):
        next_state_vec = extrapolate_state_vec + kalman_gain_mat @ (measure_vec - extrapolate_measure_vec)
        return next_state_vec
    
    @staticmethod
    def cov_update(extrapolate_cov_mat, measure_cov_mat, kalman_gain):
        next_state_cov_mat = extrapolate_cov_mat - kalman_gain @ measure_cov_mat @ kalman_gain.T
        return next_state_cov_mat