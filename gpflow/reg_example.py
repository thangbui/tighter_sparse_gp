import matplotlib.pyplot as plt
import numpy as np
import gpflow
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)

from tighter_models import TighterSGPR

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

data = np.loadtxt("snelson.csv", delimiter=",")

x_train = data[:, 0].reshape(-1, 1)
y_train = data[:, 1].reshape(-1, 1)
x_plot = np.linspace(-4, 10, 100)[:, None]


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
ind = np.random.permutation(x_train.shape[0])[:M]
z = x_train[ind].copy()
sgpr = gpflow.models.SGPR(
    (x_train, y_train),
    kernel=gpflow.kernels.SquaredExponential(),
    inducing_variable=z,
)


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


tighter_sgpr = TighterSGPR(
    (x_train, y_train),
    kernel=gpflow.kernels.SquaredExponential(),
    inducing_variable=z,
)

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


fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plot_model(sgpr, ax, "SGPR", colors[0])
plot_model(tighter_sgpr, ax, "Tighter SGPR", colors[1])
plt.legend()
plt.xlabel("x")
plt.ylabel("y")

plt.savefig(
    f"/tmp/snelson_predictions.pdf",
    bbox_inches="tight",
    pad_inches=0,
)


plt.figure(figsize=(10, 6))
plt.plot(sgpr_losses, color=colors[0], label="SGPR")
plt.plot(tighter_sgpr_losses, color=colors[1], label="Tighter SGPR")
plt.xlabel("iterations")
plt.ylabel("bounds")
plt.legend()

plt.savefig(
    f"/tmp/snelson_bounds.pdf",
    bbox_inches="tight",
    pad_inches=0,
)

plt.show()
