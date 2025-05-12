import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"

import numpy as np
import matplotlib.pyplot as plt
import gpflow
from tsvgp.gpflow.svgp import TighterSVGP
from tsvgp.gpflow.sgpr import TighterSGPR
from gpflow.models.sgpr import SGPR
from tsvgp.gpflow.svgp import SVGP
from tqdm import tqdm
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

data = np.loadtxt("snelson.csv", delimiter=",")

x_train = data[:, 0].reshape(-1, 1)
y_train = data[:, 1].reshape(-1, 1)
x_plot = np.linspace(-2, 8, 100)[:, None]

colors_dict = {
    "blue": "#377eb8",
    "orange": "#ff7f00",
    "green": "#4daf4a",
    "pink": "#f781bf",
    "brown": "#a65628",
    "purple": "#984ea3",
    "gray": "#999999",
    "red": "#e41a1c",
    "yellow": "#dede00",
}
color_names = list(colors_dict.keys())
colors = list(colors_dict.values())

def plot_model(m, ax, name="", color="b"):
    pY, pYv = m.predict_y(x_plot)
    pf, pfv = m.predict_f(x_plot)
    ax.plot(x_train, y_train, "xk")
    ax.plot(x_plot, pY, color=color, label=name)
    if not isinstance(m, gpflow.models.GPR):
        Z = m.inducing_variable.Z.numpy()
        ax.plot(Z, -2.5 * np.ones_like(Z), "o", color=color)
    two_sigma = (2.0 * pYv**0.5)[:, 0]
    ax.fill_between(
        x_plot[:, 0],
        pY[:, 0] - two_sigma,
        pY[:, 0] + two_sigma,
        alpha=0.2,
        color=color,
    )
    two_sigma = (2.0 * pfv**0.5)[:, 0]
    ax.fill_between(
        x_plot[:, 0],
        pf[:, 0] - two_sigma,
        pf[:, 0] + two_sigma,
        alpha=0.4,
        color=color,
    )

M = 5
N = x_train.shape[0]
ind = np.random.permutation(x_train.shape[0])[:M]
np.random.seed(42)
tf.random.set_seed(42)
sgpr = SGPR(
    (x_train, y_train),
    kernel=gpflow.kernels.SquaredExponential(),
    inducing_variable=x_train[ind].copy(),
)
np.random.seed(42)
tf.random.set_seed(42)
tighter_sgpr = TighterSGPR(
    (x_train, y_train),
    kernel=gpflow.kernels.SquaredExponential(),
    inducing_variable=x_train[ind].copy(),
)

np.random.seed(42)
tf.random.set_seed(42)
sgpr_losses = []
def callback():
    sgpr_losses.append(sgpr.training_loss().numpy())
execute_task = gpflow.monitor.ExecuteCallback(callback)
task_group = gpflow.monitor.MonitorTaskGroup(execute_task, period=1)
monitor = gpflow.monitor.Monitor(task_group)
opt = gpflow.optimizers.Scipy()
opt.minimize(
    sgpr.training_loss,
    sgpr.trainable_variables,
    method="L-BFGS-B",
    options=dict(maxiter=20000),
    tol=1e-11,
    step_callback=monitor,
)

np.random.seed(42)
tf.random.set_seed(42)
tighter_sgpr_losses = []
def callback():
    tighter_sgpr_losses.append(tighter_sgpr.training_loss().numpy())
execute_task = gpflow.monitor.ExecuteCallback(callback)
task_group = gpflow.monitor.MonitorTaskGroup(execute_task, period=1)
monitor = gpflow.monitor.Monitor(task_group)

opt = gpflow.optimizers.Scipy()
opt.minimize(
    tighter_sgpr.training_loss,
    tighter_sgpr.trainable_variables,
    method="L-BFGS-B",
    options=dict(maxiter=20000),
    tol=1e-11,
    step_callback=monitor,
)


np.random.seed(42)
tf.random.set_seed(42)
svgp = SVGP(
    gpflow.kernels.SquaredExponential(), 
    gpflow.likelihoods.Gaussian(), 
    x_train[ind].copy(), 
    num_data=N
)

np.random.seed(42)
tf.random.set_seed(42)
tighter_svgp = TighterSVGP(
    gpflow.kernels.SquaredExponential(), 
    gpflow.likelihoods.Gaussian(), 
    x_train[ind].copy(), 
    num_data=N
)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat().shuffle(N)
minibatch_size = N
def run_adam(model, iterations):
    np.random.seed(42)
    tf.random.set_seed(42)
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.005)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    avg_elbo = 0.0
    for step in tqdm(range(iterations)):
        optimization_step()
        avg_elbo += training_loss().numpy()
        if step > 0 and step % 10 == 0:
            logf.append(avg_elbo / 10)
            avg_elbo = 0.0
    return logf

maxiter = 10000
svgp_losses = run_adam(svgp, maxiter)
tighter_svgp_losses = run_adam(tighter_svgp, maxiter)


fig, axs = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'width_ratios': [2, 4]})

ax = axs[0, 0]
ax.plot(np.array(sgpr_losses) / N, color=colors[0], label="SGPR")
ax.plot(np.array(tighter_sgpr_losses) / N, color=colors[1], label="T-SGPR")

ax = axs[0, 1]
plot_model(sgpr, ax, "SGPR", colors[0])
plot_model(tighter_sgpr, ax, "T-SGPR", colors[1])
ax.text(
    0.45,
    0.20, 
    fr"SGPR: $\sigma^2_f$={sgpr.kernel.variance.numpy():.3f}, $l_f$={sgpr.kernel.lengthscales.numpy():.3f}, $\sigma^2$={sgpr.likelihood.variance.numpy():.3f}",
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment='top',
    color=colors[0]
)
ax.text(
    0.45,
    0.15, 
    fr"T-SGPR: $\sigma^2_f$={tighter_sgpr.kernel.variance.numpy():.3f}, $l_f$={tighter_sgpr.kernel.lengthscales.numpy():.3f}, $\sigma^2$={tighter_sgpr.likelihood.variance.numpy():.3f}",
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment='top',
    color=colors[1]
)

ax = axs[1, 0]
ax.plot(svgp_losses, color=colors[0], label="SVGP")
ax.plot(tighter_svgp_losses, color=colors[1], label="T-SVGP")

ax = axs[1, 1]
plot_model(svgp, ax, "SVGP", colors[0])
plot_model(tighter_svgp, ax, "T-SVGP", colors[1])
ax.text(
    0.45,
    0.20, 
    fr"SVGP: $\sigma^2_f$={svgp.kernel.variance.numpy():.3f}, $l_f$={svgp.kernel.lengthscales.numpy():.3f}, $\sigma^2$={svgp.likelihood.variance.numpy():.3f}",
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment='top',
    color=colors[0]
)
ax.text(
    0.45,
    0.15,
    fr"T-SVGP: $\sigma^2_f$={tighter_svgp.kernel.variance.numpy():.3f}, $l_f$={tighter_svgp.kernel.lengthscales.numpy():.3f}, $\sigma^2$={tighter_svgp.likelihood.variance.numpy():.3f}",
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment='top',
    color=colors[1]
)

axs[0, 0].set_xlabel("iteration [l-bfgs]")
axs[1, 0].set_xlabel("iteration [adam]")
axs[0, 0].set_ylabel("negative lower bound [SGPR/T-SGPR]")
axs[1, 0].set_ylabel("negative lower bound [SVGP/T-SVGP]")
axs[0, 0].legend()
axs[0, 0].set_yscale("log")
axs[1, 0].legend()
axs[1, 0].set_yscale("log")

axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("y")
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("y")

plt.savefig("snelson_results.png", bbox_inches='tight', dpi=300, pad_inches=0.05)