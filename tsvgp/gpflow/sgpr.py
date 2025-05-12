import tensorflow as tf
from check_shapes import check_shapes
from gpflow.utilities import to_default_float
from gpflow.models.sgpr import SGPR


class TighterSGPR(SGPR):
    
    @check_shapes(
        "return: []",
    )
    def logdet_term(self, common: "SGPR.CommonTensors") -> tf.Tensor:
        r"""
        Bound from Jensen's Inequality:

        .. math::
            \log |K + σ²I| <= \log |Q + σ²I| + N * \sum_n \log (1 + \textrm{diag}_n(K - Q)/σ²)

        :param common: A named tuple containing matrices that will be used
        :return: log_det, lower bound on :math:`-.5 * \textrm{output_dim} * \log |K + σ²I|`
        """
        sigma_sq = common.sigma_sq
        LB = common.LB
        A = common.A

        x, y = self.data
        outdim = to_default_float(tf.shape(y)[1])
        kdiag = self.kernel(x, full_cov=False)

        # ****************************************************
        qdiag = tf.reduce_sum(tf.square(A), 0)
        diag_term = kdiag / sigma_sq - qdiag
        correction = tf.reduce_sum(tf.math.log(1 + diag_term))
        # ****************************************************

        # 0.5 * log(det(B))
        half_logdet_b = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))

        # sum log(σ²)
        log_sigma_sq = tf.reduce_sum(tf.math.log(sigma_sq))

        logdet_k = -outdim * (
            half_logdet_b + 0.5 * log_sigma_sq + 0.5 * correction
        )
        return logdet_k
