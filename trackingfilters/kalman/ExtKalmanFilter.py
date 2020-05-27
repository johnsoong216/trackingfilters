import numpy as np
from trackingfilters.kalman.BaseKalman import BaseKalmanFilter
from trackingfilters.exceptions import ExtKalmanException


class ExtKalmanFilter(BaseKalmanFilter):

    def __init__(self, init_state_vec, init_state_cov_mat, process_noise_mat,
                 measure_noise_mat, transition_func, observation_func):

        super(ExtKalmanFilter, self).__init__(init_state_vec=init_state_vec,
                                              init_state_cov_mat=init_state_cov_mat,
                                              process_noise_mat=process_noise_mat,
                                              measure_noise_mat=measure_noise_mat)

        ExtKalmanFilter.dimension_check(
            init_state_vec.shape,
            init_state_cov_mat.shape,
            BaseKalmanFilter.mat_shape(process_noise_mat),
            BaseKalmanFilter.mat_shape(measure_noise_mat)
        )

        self.func_check(transition_func, init_state_vec, init_state_vec.shape)

        self.transition_func = transition_func
        self.observation_func = observation_func

    def extrapolate(self, cur_state_vec, cur_state_cov_mat, transition_func, observation_func, measure_noise_mat,
                    process_noise_mat):

        transition_mat = ExtKalmanFilter.jacobian_mat(transition_func, cur_state_vec)
        extrapolate_state_vec = ExtKalmanFilter.state_extrapolation(cur_state_vec, transition_mat)

        observation_mat = ExtKalmanFilter.jacobian_mat(observation_func, extrapolate_state_vec)
        extrapolate_cov_mat = ExtKalmanFilter.cov_extrapolation(cur_state_cov_mat, transition_mat, process_noise_mat)

        kalman_gain_mat = ExtKalmanFilter.calc_kalman_gain(extrapolate_cov_mat, observation_mat, measure_noise_mat)
        return extrapolate_state_vec, extrapolate_cov_mat, kalman_gain_mat

    def predict(self, extrapolate_state_vec, extrapolate_cov_mat, measure_vec, measure_noise_mat, kalman_gain_mat,
                observation_func):

        observation_mat = ExtKalmanFilter.jacobian_mat(observation_func, extrapolate_state_vec)
        next_state_vec = ExtKalmanFilter.state_update(extrapolate_state_vec, kalman_gain_mat, measure_vec, observation_func)
        next_state_cov_mat = ExtKalmanFilter.cov_update(extrapolate_cov_mat, kalman_gain_mat, observation_mat,
                                                        measure_noise_mat)
        return next_state_vec, next_state_cov_mat

    def vec_update(self, measure_vec):

        self.measure_check(measure_vec)

        for measure in measure_vec:
            if isinstance(measure, np.ndarray):
                measure = measure.reshape(-1, 1)
            self.step_update(measure)

    def step_update(self, measure_vec, cur_state_vec=None, cur_state_cov_mat=None, transition_func=None,
                    observation_func=None, control_mat=None, control_vec=None, measure_noise_mat=None,
                    process_noise_mat=None):

        self.update_external_params(cur_state_vec, cur_state_cov_mat, transition_func, observation_func, measure_noise_mat,
                                    process_noise_mat)

        self.func_check(self.observation_func, measure_vec, self.cur_state_vec)

        try:
            extrapolate_state_vec, extrapolate_cov_mat, kalman_gain_mat = self.extrapolate(
                self.cur_state_vec,
                self.cur_state_cov_mat,
                self.transition_func if type(self.transition_func) != list else self.transition_func[self.state_count],
                self.observation_func if type(self.observation_func) != list else self.observation_func[self.state_count],
                self.measure_noise_mat if self.measure_noise_mat.ndim == 2 else self.measure_noise_mat[self.state_count],
                self.process_noise_mat if self.process_noise_mat.ndim == 2 else self.process_noise_mat[self.state_count]
            )

            next_state_vec, next_state_cov_mat = self.predict(
                extrapolate_state_vec,
                extrapolate_cov_mat,
                measure_vec,
                self.measure_noise_mat if self.measure_noise_mat.ndim == 2 else self.measure_noise_mat[self.state_count],
                kalman_gain_mat,
                self.observation_func
            )

        except IndexError:
            raise ExtKalmanException("List of Matrices passed in is less than the number of observations.")

        self.record_update(kalman_gain_mat, next_state_cov_mat, next_state_vec)
        return next_state_vec, next_state_cov_mat, kalman_gain_mat

    @staticmethod
    def dimension_check(init_state_vec_s, init_state_cov_mat_s, process_noise_mat_s, measure_noise_mat_s):

        if not (init_state_vec_s[1] == 1):
            raise ExtKalmanException("Init State Vector is not in column vector format")

        if init_state_cov_mat_s != process_noise_mat_s:
            raise ExtKalmanException(
                "Init State Covariance Matrix does not have the same shape as the Process Noise Matrix")

        if init_state_vec_s[0] != init_state_cov_mat_s[0]:
            raise ExtKalmanException("Init State Covariance Matrix and Init State Vector do not have consistent shape")

    @staticmethod
    def func_check(func, input_vec, output_vec_dim):
        if func(input_vec).shape != output_vec_dim:
            raise ExtKalmanException("Function Input/Output Size has a mismatch")

    def update_external_params(self, cur_state_vec, cur_state_cov_mat, transition_func, observation_func, measure_noise_mat,
                               process_noise_mat):
        if any([x.dim > 2 for x in
                [cur_state_vec, cur_state_cov_mat, measure_noise_mat, process_noise_mat] if
                x is not None]):
            raise ExtKalmanException(
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
    def state_extrapolation(cur_state_vec, transition_func):
        """
        Function extraploates state_vec from time step t to time step t + 1, cur_state_vec(t,t), next_state_vec(t+1, t)
        Variable Dimensions:
        cur_state_vec: n * 1
        transition_mat: n * n
        """
        extrapolate_state_vec = transition_func(cur_state_vec)  # + np.dot(control_mat, control_vec)
        return extrapolate_state_vec

    @staticmethod
    def state_update(extrapolate_state_vec, kalman_gain_mat, measure_vec, observation_func):
        """
        Dimension
        Prev next state vec n * 1
        Measure vc z * 1
        Observation Z * X
        Kalman Gain X * Z
        """

        next_state_vec = extrapolate_state_vec + np.dot(kalman_gain_mat,
                                                        measure_vec - observation_func(extrapolate_state_vec))
        return next_state_vec

    @staticmethod
    def jacobian_mat(func, state_vec):
        J_mat = np.zeros(shape=(len(state_vec), len(state_vec)))

        h = 10 ** -6

        # Centred Difference Approach
        for i in range(len(state_vec)):
            vec_1 = state_vec.copy()
            vec_2 = state_vec.copy()

            vec_1[i][0] = state_vec[i][0] + h
            vec_2[i][0] = state_vec[i][0] - h

            J_mat[:, i] = ((func(vec_1) - func(vec_2)) / (2 * h)).flatten()
        return J_mat
