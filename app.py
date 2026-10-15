"""Interactive demo: compare integrators on Kepler, figure-eight and Plummer-cluster systems.

Uses the NumPy direct-sum force. Energy is always measured with the exact direct-sum potential.
Run locally with:  streamlit run app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from nbody import simulate, two_body_orbit, figure_eight, plummer_sphere

COLORS = {"euler": "C0", "verlet": "C1", "rk4": "C2"}
LIMITS = {"Kepler orbit": 2.0, "Figure-eight (3 bodies)": 1.5}

st.set_page_config(page_title="N-body integrator demo", layout="wide")
st.title("N-body integrator demo")
st.caption("From-scratch Euler, Velocity Verlet and RK4 with NumPy direct-sum gravity. "
           "Code, tests and benchmarks (Numba, Barnes-Hut): github.com/dmatharv25-cmd/nbody-sim")


@st.cache_data(show_spinner=False)
def run(system, method, steps_per_period, n_periods, ecc, n_bodies, seed):
    soft = 0.0
    if system == "Kepler orbit":
        pos, vel, m, T = two_body_orbit(e=ecc)
        dt, n_steps = T / steps_per_period, steps_per_period * n_periods
    elif system == "Figure-eight (3 bodies)":
        pos, vel, m, T = figure_eight()
        dt, n_steps = T / steps_per_period, steps_per_period * n_periods
    else:
        pos, vel, m = plummer_sphere(n_bodies, seed=seed)
        soft, dt, n_steps = 0.05, 0.01, 300
    record_every = max(1, n_steps // 600)
    with np.errstate(all="ignore"):
        out = simulate(pos, vel, m, dt, n_steps, method=method, softening=soft,
                       record_every=record_every)
    return {k: out[k] for k in ("t", "pos", "energy", "p")}


with st.sidebar:
    st.header("Settings")
    system = st.selectbox("System", ["Kepler orbit", "Figure-eight (3 bodies)",
                                     "Plummer cluster (3D)"])
    methods = st.multiselect("Integrators", ["euler", "verlet", "rk4"],
                             default=["euler", "verlet", "rk4"])
    ecc, n_bodies, seed = 0.6, 100, 0
    steps_per_period, n_periods = 200, 10
    if system == "Plummer cluster (3D)":
        n_bodies = st.slider("Number of bodies", 20, 150, 100, step=10)
        seed = st.number_input("Random seed", 0, 1000, 0)
        st.caption("Fixed: softening 0.05, dt = 0.01, 300 steps.")
    else:
        steps_per_period = st.slider("Steps per period", 20, 500, 200, step=10)
        n_periods = st.slider("Periods", 1, 20, 10)
        if system == "Kepler orbit":
            ecc = st.slider("Eccentricity", 0.0, 0.9, 0.6, step=0.05)

if not methods:
    st.info("Pick at least one integrator in the sidebar.")
    st.stop()

with st.spinner("Simulating..."):
    results = {m: run(system, m, steps_per_period, n_periods, ecc, n_bodies, seed)
               for m in methods}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
rows = []
for m in methods:
    r = results[m]
    E0 = r["energy"][0]
    err = np.abs((r["energy"] - E0) / E0)
    ax2.semilogy(r["t"], np.maximum(err, 1e-17), color=COLORS[m], label=m)
    rows.append({"integrator": m,
                 "max |dE/E|": f"{np.nanmax(err):.2e}" if np.isfinite(err).any() else "diverged",
                 "max |total momentum|": f"{np.nanmax(r['p']):.1e}"})
    if system != "Plummer cluster (3D)":
        for b in range(r["pos"].shape[1]):
            ax1.plot(r["pos"][:, b, 0], r["pos"][:, b, 1], color=COLORS[m], lw=1,
                     label=m if b == 0 else None)

if system == "Plummer cluster (3D)":
    first = results[methods[0]]["pos"]
    ax1.scatter(first[0, :, 0], first[0, :, 1], s=8, color="lightgray", label="start")
    ax1.scatter(first[-1, :, 0], first[-1, :, 1], s=8, color=COLORS[methods[0]],
                label="end (" + methods[0] + ")")
    ax1.set_title("Cluster, x-y projection")
else:
    lim = LIMITS[system]
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
    ax1.set_title("Trajectories (axes clipped)")
ax1.set_aspect("equal", adjustable="box"); ax1.legend(loc="upper right", fontsize=8)
ax1.set_xlabel("x"); ax1.set_ylabel("y")
ax2.set(xlabel="time", ylabel="|dE / E0|", title="Relative energy error")
ax2.grid(alpha=.3, which="both"); ax2.legend()
fig.tight_layout()
st.pyplot(fig)

st.table(rows)
st.caption("Euler's energy error grows steadily, Velocity Verlet's stays bounded (symplectic), "
           "and RK4 is much more accurate per step but drifts slowly. Everything here uses the exact "
           "direct-sum force; the Barnes-Hut results are in the repository README.")
