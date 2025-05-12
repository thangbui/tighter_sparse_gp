from typing import Optional

import tensorflow as tf
from check_shapes import inherit_check_shapes

from gpflow.base import InputData, MeanAndVariance, Parameter, RegressionData
from gpflow.config import default_jitter, default_float
from gpflow.inducing_variables import InducingVariables
from gpflow.kernels import Kernel
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.likelihoods import Gaussian

from gpflow.models.svgp import SVGP as GPflowSVGP

class SVGP(GPflowSVGP):
    @inherit_check_shapes
    def elbo(self, data):
        res = super().elbo(data)
        if self.num_data is None:
            num_data = tf.shape(data[0])[0]
        else:
            num_data = self.num_data
        return res / num_data


# this is the same as TighterSGPR but allows stochastic training
class TighterSVGP(SVGP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert(isinstance(self.likelihood, Gaussian))

    @inherit_check_shapes
    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X)

        D_diag = self.get_D(X)
        sigma_sq = self.likelihood.variance
        correct_term = - 0.5 * tf.reduce_sum(tf.math.log(1 + D_diag / sigma_sq))
        correct_term += 0.5 * tf.reduce_sum(D_diag / sigma_sq)
        var_exp = self.likelihood.variational_expectations(X, f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
            num_data = tf.shape(X)[0]
        res = tf.reduce_sum(var_exp) * scale - kl + correct_term * scale
        res = res / num_data
        return res

    def get_D(self, X):
        Kmm = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        Kmn = Kuf(self.inducing_variable, self.kernel, X)
        Knn_diag = self.kernel(X, full_cov=False)

        # compute kernel stuff
        K = tf.rank(Kmn)
        perm = tf.concat([tf.reshape(tf.range(1, K-1), [K-2]), # leading dims (...)
                        tf.reshape(0, [1]),  # [M]
                        tf.reshape(K-1, [1])], 0)  # [N]
        Kmn = tf.transpose(Kmn, perm)  # ... x M x N
        leading_dims = tf.shape(Kmn)[:-2]
        Lm = tf.linalg.cholesky(Kmm)  # [M,M]
        # Compute the projection matrix A
        Lm = tf.broadcast_to(Lm, tf.concat([leading_dims, tf.shape(Lm)], 0))  # [...,M,M]
        A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [...,M,N]
        D_diag = Knn_diag - tf.reduce_sum(tf.square(A), -2)
        return D_diag
