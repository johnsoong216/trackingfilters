import numpy as np
from trackingfilters.exceptions import KalmanException


class BaseKalmanFilter:

    def __init__(self, init_state_vec, init_state_cov_mat, process_noise_mat, measure_noise_mat):

        # Initialize Universal Variables
        self.cur_state_vec = init_state_vec
        self.cur_state_cov_mat = init_state_cov_mat
        self.process_noise_mat = process_noise_mat
        self.measure_noise_mat = measure_noise_mat

        # Initialize aggregated data
        self.state_count = 0
        self.agg_state_vec = init_state_vec.reshape(1, init_state_vec.shape[0], init_state_vec.shape[1])
        self.agg_state_cov_mat = init_state_cov_mat.reshape(1, init_state_cov_mat.shape[0], init_state_cov_mat.shape[1])
        self.agg_kalman_gain_mat = np.zeros(shape=(1, init_state_cov_mat.shape[0], measure_noise_mat.shape[1]))

    def agg_result(self):
        return self.agg_state_vec, self.agg_state_cov_mat, self.agg_kalman_gain_mat

    def reset(self):
        self.state_count = 0
        self.agg_state_vec = self.agg_state_vec[:1]
        self.agg_state_cov_mat = self.agg_state_cov_mat[:1]
        self.agg_kalman_gain_mat = self.agg_kalman_gain_mat[:1]

    def record_update(self, kalman_gain_mat, next_state_cov_mat, next_state_vec):
        self.cur_state_vec = next_state_vec
        self.cur_state_cov_mat = next_state_cov_mat
        self.state_count += 1

        # print(next_state_vec)
        # print(next_state_cov_mat)

        self.agg_state_vec = np.concatenate(
            [self.agg_state_vec, next_state_vec.reshape(1, next_state_vec.shape[0], next_state_vec.shape[1])])
        self.agg_state_cov_mat = np.concatenate([self.agg_state_cov_mat,
                                                 next_state_cov_mat.reshape(1, next_state_cov_mat.shape[0],
                                                                            next_state_cov_mat.shape[1])])
        self.agg_kalman_gain_mat = np.concatenate(
            [self.agg_kalman_gain_mat, kalman_gain_mat.reshape(1, kalman_gain_mat.shape[0], kalman_gain_mat.shape[1])])

    def extrapolate(self, **kwargs):
        raise NotImplementedError

    def predict(self, **kwargs):
        raise NotImplementedError

    def vec_update(self, measure_vec):
        raise NotImplementedError

    def step_update(self, measure, **kwargs):
        raise NotImplementedError

    def update_external_params(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def state_extrapolation(**kwargs):
        raise NotImplementedError

    @staticmethod
    def state_update(**kwargs):
        raise NotImplementedError

    @staticmethod
    def cov_extrapolation(**kwargs):
        raise NotImplementedError

    @staticmethod
    def calc_kalman_gain(**kwargs):
        raise NotImplementedError

    @staticmethod
    def cov_update(**kwargs):
        raise NotImplementedError

    @staticmethod
    def measure_extrapolation(**kwargs):
        raise NotImplementedError

    # Helper Function
    @staticmethod
    def dimension_check(**kwargs):
        raise NotImplementedError

    @staticmethod
    def measure_check(measure_vec):
        if measure_vec.ndim < 2 and isinstance(measure_vec[0], np.ndarray):
            raise KalmanException(
                'Please call method stepwise_update if measurements are not in 2-dimensional array format')

    @staticmethod
    def mat_shape(mat):
        return mat.shape if mat.ndim == 2 else mat[0].shape

