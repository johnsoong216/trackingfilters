import numpy as np
from scipy import integrate, linalg
from trackingfilters.exceptions import LinKalmanException

class LinKalmanFilter:

    def __init__(self, init_state_vec, init_state_cov_mat, transition_mat, observation_mat, control_vec, control_mat,
                 process_noise_mat, measure_noise_mat):

        self.dimension_check(
            init_state_vec.shape,
            init_state_cov_mat.shape,
            transition_mat.shape if transition_mat.ndim == 2 else transition_mat[0].shape,
            observation_mat.shape if observation_mat.ndim == 2 else observation_mat[0].shape,
            control_vec.shape if control_vec.ndim == 2 else control_vec[0].shape,
            control_mat.shape if control_mat.ndim == 2 else control_mat[0].shape,
            process_noise_mat.shape if process_noise_mat.ndim == 2 else process_noise_mat[0].shape,
            measure_noise_mat.shape if measure_noise_mat.ndim == 2 else process_noise_mat[0].shape
        )

        self.cur_state_vec = init_state_vec
        self.cur_state_cov_mat = init_state_cov_mat
        self.transition_mat = transition_mat
        self.observation_mat = observation_mat
        self.control_vec = control_vec
        self.control_mat = control_mat
        self.process_noise_mat = process_noise_mat
        self.measure_noise_mat = measure_noise_mat

        self.state_count = 0
        self.agg_state_vec = init_state_vec.reshape(1, init_state_vec.shape[0], init_state_vec.shape[1])
        self.agg_state_cov_mat = init_state_cov_mat.reshape(1, init_state_cov_mat.shape[0], init_state_cov_mat.shape[1])
        self.agg_kalman_gain_mat = np.zeros(shape=(1, init_state_cov_mat.shape[0], measure_noise_mat.shape[1]))


    def extrapolate(self, cur_state_vec, cur_state_cov_mat, transition_mat, observation_mat, control_mat, control_vec, measure_noise_mat, process_noise_mat):

        extrapolate_state_vec = LinKalmanFilter.state_extrapolation(cur_state_vec, control_vec, transition_mat, control_mat)
        extrapolate_cov_mat = LinKalmanFilter.cov_extrapolation(cur_state_cov_mat, transition_mat, process_noise_mat)
        kalman_gain_mat = LinKalmanFilter.calc_kalman_gain(extrapolate_cov_mat, observation_mat, measure_noise_mat)
        return extrapolate_state_vec, extrapolate_cov_mat, kalman_gain_mat

    def predict(self, extrapolate_state_vec, extrapolate_cov_mat, measure_vec, measure_noise_mat, kalman_gain_mat,
               observation_mat):

        next_state_vec = LinKalmanFilter.state_update(extrapolate_state_vec, kalman_gain_mat, measure_vec, observation_mat)
        next_state_cov_mat = LinKalmanFilter.cov_update(extrapolate_cov_mat, kalman_gain_mat, observation_mat, measure_noise_mat)
        return next_state_vec, next_state_cov_mat

    def vec_update(self, measure_vec):

        if measure_vec.ndim < 2 and isinstance(measure_vec[0], np.ndarray):
            raise LinKalmanException('Please call method stepwise_update if measurements are not in 2-dimensional array format')
        # if measure_vec[0].reshape(-1, 1) != measure_vec[0]:
        #     raise LinKalmanException('Measure vector must be in column vector format')

        for measure in measure_vec:
            if isinstance(measure, np.ndarray):
                measure = measure.reshape(-1, 1)
            self.step_update(measure)


    def step_update(self, measure_vec, cur_state_vec=None, cur_state_cov_mat=None, transition_mat=None, observation_mat=None, control_mat=None, control_vec=None, measure_noise_mat=None, process_noise_mat=None):

        self.update_external_params(cur_state_vec, cur_state_cov_mat, transition_mat, observation_mat, control_mat, control_vec, measure_noise_mat, process_noise_mat)

        try:
            extrapolate_state_vec, extrapolate_cov_mat, kalman_gain_mat = self.extrapolate(
            self.cur_state_vec,
            self.cur_state_cov_mat,
            self.transition_mat if self.transition_mat.ndim == 2 else self.transition_mat[self.state_count],
            self.observation_mat if self.observation_mat.ndim == 2 else self.observation_mat[self.state_count],
            self.control_mat if self.control_mat.ndim == 2 else self.control_mat[self.state_count],
            self.control_vec if self.control_vec.ndim == 2 else self.control_vec[self.state_count],
            self.measure_noise_mat if self.measure_noise_mat.ndim == 2 else self.measure_noise_mat[self.state_count],
            self.process_noise_mat if self.process_noise_mat.ndim == 2 else self.process_noise_mat[self.state_count]
            )

            next_state_vec, next_state_cov_mat = self.predict(
            extrapolate_state_vec,
            extrapolate_cov_mat,
            measure_vec,
            self.measure_noise_mat if self.measure_noise_mat.ndim == 2 else self.measure_noise_mat[self.state_count],
            kalman_gain_mat,
            self.observation_mat if self.observation_mat.ndim == 2 else self.observation_mat[self.state_count]
            )

        except IndexError:
            raise LinKalmanException("List of Matrices passed in is less than the number of observations.")


        self.cur_state_vec = next_state_vec
        self.cur_state_cov_mat = next_state_cov_mat
        self.state_count += 1

        #
        # print(self.agg_state_vec)
        # print(next_state_vec)
        # print(next_state_vec.reshape(1, next_state_vec.shape[0], next_state_vec.shape[1]))


        self.agg_state_vec = np.concatenate([self.agg_state_vec, next_state_vec.reshape(1, next_state_vec.shape[0], next_state_vec.shape[1])])
        self.agg_state_cov_mat = np.concatenate([self.agg_state_cov_mat, next_state_cov_mat.reshape(1, next_state_cov_mat.shape[0], next_state_cov_mat.shape[1])])
        self.agg_kalman_gain_mat = np.concatenate([self.agg_kalman_gain_mat, kalman_gain_mat.reshape(1, kalman_gain_mat.shape[0], kalman_gain_mat.shape[1])])

        return next_state_vec, next_state_cov_mat, kalman_gain_mat

    def agg_result(self):
        return self.agg_state_vec, self.agg_state_cov_mat, self.agg_kalman_gain_mat

    def dimension_check(self, init_state_vec_s,
                              init_state_cov_mat_s,
                              transition_mat_s,
                              observation_mat_s,
                              control_vec_s,
                              control_mat_s,
                              process_noise_mat_s,
                              measure_noise_mat_s):

        if not (init_state_vec_s[1] == 1 and control_vec_s[1] == 1):
            raise LinKalmanException("Init State Vector or Control Vector are not in column vector format")

        if (init_state_cov_mat_s != process_noise_mat_s) or (init_state_cov_mat_s != transition_mat_s):
            raise LinKalmanException("Init State Covariance Matrix does not have the same shape as the Process Noise Matrix or Transition Matrix")

        if init_state_vec_s[0] != init_state_cov_mat_s[0]:
            raise LinKalmanException("Init State Covariance Matrix and Init State Vector do not have consistent shape")

        if control_mat_s[0] != init_state_vec_s[0]:
            raise LinKalmanException("Control Matrix and Init State Vector do not have consistent shape")

        if control_mat_s[1] != control_vec_s[0]:
            raise LinKalmanException("Control Matrix and Control Vector do not have consistent shape")

        if observation_mat_s[0] != measure_noise_mat_s[0]:
            raise LinKalmanException("Observation matrix and measurement noise matrix do not have consistent shape")

        if observation_mat_s[1] != init_state_vec_s[0]:
            raise LinKalmanException("Observation matrix and Init State Vector do not have consistent shape")

    def update_external_params(self, cur_state_vec, cur_state_cov_mat, transition_mat, observation_mat, control_mat, control_vec, measure_noise_mat, process_noise_mat):

        if any([x.dim > 2 for x in [cur_state_vec, cur_state_cov_mat, transition_mat, observation_mat, control_mat, control_vec, measure_noise_mat, process_noise_mat] if x is not None]):
            raise LinKalmanException("Filter does not support inputting a list of matrices/vecs when doing step-wise update")

        if cur_state_vec is not None:
            self.cur_state_vec = cur_state_vec
        if cur_state_cov_mat is not None:
            self.cur_state_cov_mat = cur_state_cov_mat
        if transition_mat is not None:
            self.transition_mat = transition_mat
        if observation_mat is not None:
            self.observation_mat = observation_mat
        if control_mat is not None:
            self.control_mat = control_mat
        if control_vec is not None:
            self.control_vec = control_vec
        if measure_noise_mat is not None:
            self.measure_noise_mat = measure_noise_mat
        if process_noise_mat is not None:
            self.process_noise_mat = process_noise_mat

    def reset(self):
        self.state_count = 0
        self.agg_state_vec = self.agg_state_vec[:1]
        self.agg_state_cov_mat = self.agg_state_cov_mat[:1]
        self.agg_kalman_gain_mat = self.agg_kalman_gain_mat[:1]


    @staticmethod
    def state_extrapolation(cur_state_vec, control_vec, transition_mat, control_mat):
        """
        Function extraploates state_vec from time step t to time step t + 1, cur_state_vec(t,t), next_state_vec(t+1, t)
        Variable Dimensions:
        cur_state_vec: n * 1
        transition_mat: n * n
        control_mat: n * m
        control_vec: m * 1
        """
        extrapolate_state_vec = np.dot(transition_mat, cur_state_vec) + np.dot(control_mat, control_vec)
        return extrapolate_state_vec

    @staticmethod
    def cov_extrapolation(cur_cov_mat, transition_mat, process_noise_mat):
        extrapolate_cov_mat = np.linalg.multi_dot([transition_mat, cur_cov_mat, transition_mat.transpose()]) + \
                       process_noise_mat
        return extrapolate_cov_mat

    @staticmethod
    def calc_kalman_gain(extrapolate_cov_mat, observation_mat, measure_noise_mat):
        """
        observation_mat Z * X
        extrapolate_cov_mat  X * X
        measure_noise_mat Z * Z
        """

        kalman_gain_mat = np.linalg.multi_dot([
            extrapolate_cov_mat,
            observation_mat.transpose(),
            np.linalg.inv(
                np.linalg.multi_dot(
                    [observation_mat, extrapolate_cov_mat, observation_mat.transpose()]) + \
                measure_noise_mat
            )]
        )

        return kalman_gain_mat

    @staticmethod
    def cov_update(extrapolate_cov_mat, kalman_gain_mat, observation_mat, measure_noise_mat):
        """
        Kalman Gain X * Z
        Observation Z * X
        Identity X * X
        Previous Cov Mat  X * X
        Measure Noise Z * 1

        """

        identity_mat = np.eye(kalman_gain_mat.shape[0])

        first_part = np.subtract(identity_mat, np.dot(kalman_gain_mat, observation_mat))
        next_state_cov_mat = np.linalg.multi_dot([first_part, extrapolate_cov_mat, first_part.transpose()]) + \
                       np.linalg.multi_dot([kalman_gain_mat, measure_noise_mat, kalman_gain_mat.transpose()])

        return next_state_cov_mat

    @staticmethod
    def state_update(extrapolate_state_vec, kalman_gain_mat, measure_vec, observation_mat):
        """
        Dimension
        Prev next state vec n * 1
        Measure vc z * 1
        Observation Z * X
        Kalman Gain X * Z
        """

        next_state_vec = extrapolate_state_vec + np.dot(kalman_gain_mat,
                                                        measure_vec - np.dot(observation_mat, extrapolate_state_vec))
        return next_state_vec


    ### Helper Functions to Initialize Matrices
    @staticmethod
    def gen_transition_control_mat(sys_mat, input_mat, delta_t):

        transition_mat = linalg.expm(sys_mat * delta_t)

        control_mat = np.zeros(shape=transition_mat.shape)
        for row in range(sys_mat.shape[0]):
            for col in range(sys_mat.shape[1]):
                control_mat[row][col] = integrate.quad(LinKalmanFilter.sys_mat_integral, 0, delta_t, args=(sys_mat, row, col))[0]

        control_mat = np.dot(control_mat, input_mat)
        return transition_mat, control_mat

    @staticmethod
    def noise_integral(process_noise_func, delta_t, control_mat_func, process_noise_vec, dim_x, dim_y):
        return integrate.quad(func=process_noise_func, a=0, b=delta_t,
                              args=(control_mat_func, process_noise_vec, dim_x, dim_y))[0]
    @staticmethod
    def sys_mat_integral(delta_t, sys_mat, dim_x, dim_y):
        return linalg.expm(sys_mat * delta_t)[dim_x][dim_y]

    @staticmethod
    def calc_process_noise_mat(process_noise_vec, control_mat, time_dependence=False, delta_t=0, continuous=False,
                               control_mat_func=None):
        if not time_dependence:
            process_noise_mat = np.diag(process_noise_vec)

        else:
            if not continuous:
                process_noise_mat = np.linalg.multi_dot([np.diag(process_noise_vec), control_mat, control_mat.transpose()])
            else:
                dim = LinKalmanFilter.process_noise_func(delta_t, control_mat_func, process_noise_vec).shape

                process_noise_mat = np.zeros(shape=dim)

                for row in range(dim[0]):
                    for col in range(dim[1]):
                        process_noise_mat[row][col] = LinKalmanFilter.noise_integral(LinKalmanFilter.process_noise_func, delta_t, control_mat_func,
                                                                     process_noise_vec, row, col)

        return process_noise_mat

    @staticmethod
    def process_noise_func(delta_t, control_mat_func, process_noise_vec, dim_x=None, dim_y=None):
        process_noise_mat = np.linalg.multi_dot(
            [np.diag(process_noise_vec), control_mat_func(delta_t), control_mat_func(delta_t).transpose()]
        )
        if dim_x is None and dim_y is None:
            return process_noise_mat
        return process_noise_mat[dim_x][dim_y]

    @staticmethod
    def measure_update(cur_state_vec, observation_mat, measure_noise_vec):
        """
        Dimension:
        State Vec n * 1
        Observation Matrix z * n
        Noise Vec z * 1
        """
        measure_vec = np.dot(observation_mat, cur_state_vec) + measure_noise_vec
        return measure_vec




