"""Generate README animations: figure-eight three-body orbit and a Plummer star cluster."""
import sys; sys.path.insert(0, ".")
from functools import partial
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from nbody import simulate, figure_eight, plummer_sphere, accelerations


def save(fig, update, n_frames, path, fps=20):
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=70)
    plt.close(fig)
    print("saved", path)


# ---- 1) figure-eight orbit (RK4), body trails ----
pos, vel, m, T = figure_eight()
n_steps = 2000
res = simulate(pos, vel, m, 2 * T / n_steps, n_steps, "rk4")
traj = res["pos"]                                   # (steps+1, 3, 2)
frames = np.linspace(0, n_steps, 100).astype(int)
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.set(xlim=(-1.4, 1.4), ylim=(-0.7, 0.7), aspect="equal", xticks=[], yticks=[])
ax.set_title("Figure-eight three-body orbit (RK4)", fontsize=9)
colors = ["tab:blue", "tab:orange", "tab:green"]
trails = [ax.plot([], [], color=c, lw=0.8, alpha=0.6)[0] for c in colors]
dots = [ax.plot([], [], "o", color=c, ms=7)[0] for c in colors]


def upd8(f):
    k = frames[f]
    for b in range(3):
        trails[b].set_data(traj[max(0, k - 400):k + 1, b, 0], traj[max(0, k - 400):k + 1, b, 1])
        dots[b].set_data([traj[k, b, 0]], [traj[k, b, 1]])
    return trails + dots


save(fig, upd8, len(frames), "results/figure_eight.gif")

# ---- 2) Plummer cluster (direct sum, Velocity Verlet), projected to x-y ----
eps, dt, steps = 0.05, 0.01, 400
p0, v0, mm = plummer_sphere(200, seed=1)
res = simulate(p0, v0, mm, dt, steps, "verlet", softening=eps, record_every=4,
               acc_fn=partial(accelerations, softening=eps))
traj = res["pos"]
fig, ax = plt.subplots(figsize=(4, 4))
ax.set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal", xticks=[], yticks=[])
ax.set_facecolor("black")
ax.set_title("Plummer star cluster, N = 200 (Verlet)", fontsize=9)
sc = ax.scatter(traj[0, :, 0], traj[0, :, 1], s=6, color="white")


def upd8c(f):
    sc.set_offsets(traj[f, :, :2])
    return (sc,)


save(fig, upd8c, len(traj), "results/cluster.gif")
