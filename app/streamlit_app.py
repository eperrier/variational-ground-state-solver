from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

from psitest.grid import GridConfig, make_grid
from psitest.potentials import PotentialConfig, available_potentials, build_potential
from psitest.models import ModelConfig, available_models, build_model, count_parameters
from psitest.solver_fd import fd_eigensolve_k
from psitest.trainer import TrainConfig, train_wavefunction, train_states_sequential, random_sweep, compare_states_to_reference
from psitest.physics import normalize_psi
from psitest.plotting import (
    plot_potential,
    plot_training,
    plot_boundary_mass,
    plot_wavefunctions,
    plot_overlap_matrix,
    save_fig,
)
from psitest.trace import RunContext, get_output_base_dir


# ---------- App config ----------
st.set_page_config(page_title="PSITest WebApp", layout="wide")

# Use double precision by default (more stable for physics). User can override per-run.
DEFAULT_DTYPE = torch.float64


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sidebar_runtime_info():
    st.sidebar.markdown("### Runtime")
    device = get_device()
    st.sidebar.write("Device:", str(device))
    if device.type == "cuda":
        st.sidebar.write("GPU:", torch.cuda.get_device_name(0))
        st.sidebar.write("CUDA:", torch.version.cuda)
    st.sidebar.write("Torch:", torch.__version__)


def _jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy/torch scalars into plain python for JSON."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif torch.is_tensor(v):
            out[k] = v.detach().cpu().tolist()
        else:
            out[k] = v
    return out


# ---------- UI helpers: potential params ----------

def potential_param_widgets(key: str, ui_key_prefix: str) -> Dict[str, float]:
    """Return params dict from widgets for a selected potential.

    NOTE: We include a ui_key_prefix so the same widgets can be used in multiple
    tabs/modules without StreamlitDuplicateElementId collisions.
    """
    params: Dict[str, float] = {}

    if key == "harmonic":
        params["omega"] = float(
            st.slider(
                "omega",
                min_value=0.25,
                max_value=4.0,
                value=1.0,
                step=0.25,
                key=f"{ui_key_prefix}_omega",
            )
        )

    if key == "double_well":
        params["a"] = float(
            st.slider(
                "a",
                min_value=0.0,
                max_value=6.0,
                value=2.0,
                step=0.25,
                key=f"{ui_key_prefix}_a",
            )
        )

    if key == "quartic_family":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            params["alpha"] = float(
                st.number_input(
                    "alpha (x^4)",
                    value=1.0,
                    step=0.1,
                    key=f"{ui_key_prefix}_alpha",
                )
            )
        with c2:
            params["beta"] = float(
                st.number_input(
                    "beta (x^2)",
                    value=-2.0,
                    step=0.1,
                    key=f"{ui_key_prefix}_beta",
                )
            )
        with c3:
            params["gamma"] = float(
                st.number_input(
                    "gamma (x)",
                    value=2.0,
                    step=0.1,
                    key=f"{ui_key_prefix}_gamma",
                )
            )
        with c4:
            params["delta"] = float(
                st.number_input(
                    "delta",
                    value=0.0,
                    step=0.1,
                    key=f"{ui_key_prefix}_delta",
                )
            )

    return params


# ---------- UI helpers: model params ----------

def model_param_widgets(key: str, ui_key_prefix: str) -> Dict[str, Any]:
    """Return params dict from widgets for a selected model architecture.

    NOTE: We include a ui_key_prefix so the same widgets can be used in multiple
    tabs/modules without StreamlitDuplicateElementId collisions.
    """
    params: Dict[str, Any] = {}

    common = st.container()
    with common:
        c1, c2, c3 = st.columns(3)
        with c1:
            params["hidden"] = int(
                st.selectbox(
                    "hidden",
                    options=[32, 64, 128, 256],
                    index=1,
                    key=f"{ui_key_prefix}_hidden",
                )
            )
        with c2:
            params["depth"] = int(
                st.selectbox(
                    "depth",
                    options=[2, 3, 4, 5],
                    index=1,
                    key=f"{ui_key_prefix}_depth",
                )
            )
        with c3:
            params["a"] = float(
                st.slider(
                    "envelope a",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.35,
                    step=0.05,
                    key=f"{ui_key_prefix}_env_a",
                )
            )

    if key in ["plain_mlp", "fourier_mlp"]:
        params["act"] = st.selectbox(
            "activation",
            options=["tanh", "silu", "gelu"],
            index=0,
            key=f"{ui_key_prefix}_act",
        )

    if key == "fourier_mlp":
        c1, c2, c3 = st.columns(3)
        with c1:
            params["num_fourier"] = int(
                st.selectbox(
                    "num_fourier",
                    options=[8, 16, 24, 32, 48],
                    index=2,
                    key=f"{ui_key_prefix}_num_fourier",
                )
            )
        with c2:
            params["ff_scale"] = float(
                st.slider(
                    "ff_scale",
                    min_value=0.5,
                    max_value=6.0,
                    value=2.5,
                    step=0.25,
                    key=f"{ui_key_prefix}_ff_scale",
                )
            )
        with c3:
            params["include_x"] = int(
                st.selectbox(
                    "include_x",
                    options=[0, 1],
                    index=1,
                    key=f"{ui_key_prefix}_include_x",
                )
            )

    if key == "siren":
        params["omega0"] = float(
            st.slider(
                "omega0",
                min_value=5.0,
                max_value=60.0,
                value=30.0,
                step=5.0,
                key=f"{ui_key_prefix}_omega0",
            )
        )

    if key == "rbf":
        c1, c2, c3 = st.columns(3)
        with c1:
            params["M"] = int(
                st.selectbox(
                    "num RBFs M",
                    options=[16, 32, 48, 64, 96],
                    index=2,
                    key=f"{ui_key_prefix}_rbf_M",
                )
            )
        with c2:
            params["init_span"] = float(
                st.slider(
                    "init_span",
                    min_value=1.0,
                    max_value=6.0,
                    value=3.0,
                    step=0.25,
                    key=f"{ui_key_prefix}_rbf_init_span",
                )
            )
        with c3:
            params["init_sigma"] = float(
                st.slider(
                    "init_sigma",
                    min_value=0.2,
                    max_value=2.0,
                    value=0.7,
                    step=0.05,
                    key=f"{ui_key_prefix}_rbf_init_sigma",
                )
            )

    return params


def train_config_widgets(ui_key_prefix: str, for_excited: bool = False) -> TrainConfig:
    """Widgets for training hyperparameters.

    NOTE: We include a ui_key_prefix so the same widgets can be used in multiple
    tabs/modules without StreamlitDuplicateElementId collisions.
    """
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        steps = int(
            st.number_input(
                "steps",
                min_value=100,
                max_value=200000,
                value=4000,
                step=500,
                key=f"{ui_key_prefix}_steps",
            )
        )
    with c2:
        lr = float(
            st.number_input(
                "lr",
                min_value=1e-5,
                max_value=1e-1,
                value=2e-3,
                step=1e-3,
                format="%.6f",
                key=f"{ui_key_prefix}_lr",
            )
        )
    with c3:
        weight_decay = float(
            st.number_input(
                "weight_decay",
                min_value=0.0,
                max_value=1e-2,
                value=1e-6,
                step=1e-6,
                format="%.8f",
                key=f"{ui_key_prefix}_weight_decay",
            )
        )
    with c4:
        grad_clip = float(
            st.number_input(
                "grad_clip",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
                key=f"{ui_key_prefix}_grad_clip",
            )
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        use_lbfgs = st.checkbox("LBFGS refinement", value=False, key=f"{ui_key_prefix}_use_lbfgs")
    with c2:
        lbfgs_steps = int(
            st.number_input(
                "lbfgs_steps",
                min_value=10,
                max_value=2000,
                value=250,
                step=50,
                disabled=not use_lbfgs,
                key=f"{ui_key_prefix}_lbfgs_steps",
            )
        )
    with c3:
        lam_ortho = float(
            st.number_input(
                "lam_ortho",
                min_value=0.0,
                max_value=500.0,
                value=80.0 if for_excited else 0.0,
                step=10.0,
                key=f"{ui_key_prefix}_lam_ortho",
            )
        )

    return TrainConfig(
        steps=steps,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        print_every=max(50, steps // 16),
        ui_update_every=max(5, steps // 200),
        ckpt_every=max(500, steps // 2),
        use_lbfgs=bool(use_lbfgs),
        lbfgs_steps=lbfgs_steps,
        lam_ortho=lam_ortho,
    )


def save_run_artifacts_ground(ctx: RunContext, x_np: np.ndarray, V_np: np.ndarray, ref: Optional[Dict[str, Any]], result: Dict[str, Any]):
    """Persist standard artifacts for a ground-state run."""

    # Training log
    df: pd.DataFrame = result["df"]
    df.to_csv(ctx.logs_dir / "training_log.csv", index=False)

    # Arrays
    psi = result["psi"].detach().cpu().numpy()
    np.savez(ctx.data_dir / "final_results.npz", x=x_np, V=V_np, psi=psi, E=result["E"], K=result["K"], U=result["U"], boundary_mass=result["boundary_mass"])

    if ref is not None:
        np.savez(ctx.data_dir / "reference_fd.npz", x=x_np, V=V_np, evals=ref["evals"], psis=ref["psis"])

    # Figures
    fig = plot_potential(x_np, V_np, title="Potential V(x)")
    save_fig(fig, ctx.figures_dir, "potential.png")

    fig = plot_training(df, title="Training curves")
    save_fig(fig, ctx.figures_dir, "training_curves.png")

    fig = plot_boundary_mass(df, title="Boundary mass")
    save_fig(fig, ctx.figures_dir, "boundary_mass.png")


def main():
    sidebar_runtime_info()

    st.title("PSITest WebApp")
    st.caption("Variational and finite-difference solvers for 1D Schrödinger problems.")

    base_dir = get_output_base_dir()
    st.sidebar.markdown("### Output")
    st.sidebar.write("Base directory:", str(base_dir))
    st.sidebar.write("(Mount ./outputs to this path via docker-compose to persist results.)")

    tabs = st.tabs([
        "Ground state",
        "Excited states",
        "Hyperparameter sweep",
        "Eigensolver",
        "Trace",
    ])

    # -------------------- Ground state --------------------
    with tabs[0]:
        st.header("Ground state (variational)")

        pot_labels = available_potentials()
        pot_key = st.selectbox("Potential", options=list(pot_labels.keys()), format_func=lambda k: pot_labels[k])
        pot_params = potential_param_widgets(pot_key, ui_key_prefix="ground_pot")

        V_torch, V_numpy, L_suggested, pot_params_full = build_potential(PotentialConfig(key=pot_key, params=pot_params))

        st.info(f"Suggested L for this potential: {L_suggested:.2f}")

        c1, c2, c3 = st.columns(3)
        with c1:
            use_suggested_L = st.checkbox("Use suggested L", value=True)
        with c2:
            L = float(st.number_input("L (domain half-width)", value=float(L_suggested), step=0.5, disabled=use_suggested_L))
        with c3:
            N = int(st.selectbox("N (grid points)", options=[1024, 2048, 4096, 8192], index=2))

        if use_suggested_L:
            L = float(L_suggested)

        model_labels = available_models()
        model_key = st.selectbox("Architecture / ansatz", options=list(model_labels.keys()), format_func=lambda k: model_labels[k])
        model_params = model_param_widgets(model_key, ui_key_prefix="ground_model")

        st.markdown("#### Training")
        cfg = train_config_widgets(ui_key_prefix="ground_train", for_excited=False)

        c1, c2, c3 = st.columns(3)
        with c1:
            compute_reference = st.checkbox("Compute FD reference (k=6)", value=True)
        with c2:
            dtype_choice = st.selectbox("dtype", options=["float64", "float32"], index=0)
        with c3:
            tag = st.text_input("run tag", value=f"ground_{pot_key}_{model_key}")

        if st.button("Run ground-state experiment", type="primary"):
            dtype = torch.float64 if dtype_choice == "float64" else torch.float32
            torch.set_default_dtype(dtype)

            device = get_device()
            x_np, x, dx = make_grid(GridConfig(L=L, N=N), device=device, dtype=dtype)
            V_np = V_numpy(x_np)
            Vx = V_torch(x)

            # Create run context + trace
            ctx = RunContext.create(kind="ground", tag=tag)
            ctx.save_env()
            ctx.snapshot_repo()

            config = {
                "experiment_kind": "ground",
                "potential": {"key": pot_key, "params": pot_params_full},
                "grid": {"L": L, "N": N, "dx": dx},
                "model": {"key": model_key, "params": model_params},
                "train": cfg.__dict__,
                "dtype": dtype_choice,
                "device": str(device),
            }
            ctx.save_config(config)

            # Potential plot in UI
            st.pyplot(plot_potential(x_np, V_np, title="Potential"), clear_figure=True)

            ref: Optional[Dict[str, Any]] = None
            if compute_reference:
                with st.spinner("Computing finite-difference reference (eigsh)..."):
                    evals, psis = fd_eigensolve_k(x_np, V_np, dx, k=6)
                    ref = {"evals": evals, "psis": psis}
                st.success(f"FD reference E0 = {evals[0]:.10f}")

            # Build + train model
            model_cfg = ModelConfig(key=model_key, params=model_params)
            model = build_model(model_cfg, L=L, positive=True).to(device)
            st.write("Trainable parameters:", count_parameters(model))

            progress_bar = st.progress(0)
            status = st.empty()

            def progress(step: int, row: Dict[str, float]):
                frac = min(1.0, step / float(cfg.steps))
                progress_bar.progress(int(100 * frac))
                status.write({
                    "step": int(step),
                    "E": row["E"],
                    "loss": row["loss"],
                    "boundary_mass": row["boundary_mass"],
                })

            def ckpt(step: int, m: torch.nn.Module, meta: Dict[str, Any]):
                torch.save({"step": step, "model_state": m.state_dict(), **meta}, ctx.ckpt_dir / f"checkpoint_step_{step:06d}.pt")

            with st.spinner("Training variational model..."):
                result = train_wavefunction(model, x, Vx, dx, cfg, orthonormal_to=None, progress=progress, checkpoint_fn=ckpt, tag="ground")

            # Persist outputs
            save_run_artifacts_ground(ctx, x_np, V_np, ref, result)

            st.success(f"Final variational energy E = {result['E']:.10f}")
            if ref is not None:
                st.write("Energy gap (E - E_ref0):", float(result["E"] - ref["evals"][0]))

            # Display training curves
            st.pyplot(plot_training(result["df"], title="Training curves"), clear_figure=True)
            st.pyplot(plot_boundary_mass(result["df"], title="Boundary mass"), clear_figure=True)

            # Wavefunction plots
            psi_nn = result["psi"].detach().cpu().numpy()
            psis_to_plot = [psi_nn]
            labels = ["psi_nn"]
            if ref is not None:
                psis_to_plot.append(ref["psis"][0])
                labels.append("psi_ref0")

            fig = plot_wavefunctions(x_np, np.stack(psis_to_plot, axis=0), labels=labels, title="Wavefunction comparison")
            st.pyplot(fig, clear_figure=True)

            # Save final model
            torch.save({"model_state": result["model"].state_dict(), "config": config}, ctx.ckpt_dir / "model_final.pt")

            # Download run zip
            zip_path = ctx.zip_run()
            st.write("Run directory:", str(ctx.run_dir))
            st.write("Run zip:", str(zip_path))
            with open(zip_path, "rb") as f:
                st.download_button("Download this run (zip)", data=f, file_name=zip_path.name, mime="application/zip")

    # -------------------- Excited states --------------------
    with tabs[1]:
        st.header("Excited states")

        pot_labels = available_potentials()
        pot_key = st.selectbox("Potential", options=list(pot_labels.keys()), format_func=lambda k: pot_labels[k], key="exc_pot")
        pot_params = potential_param_widgets(pot_key, ui_key_prefix="exc_pot")

        V_torch, V_numpy, L_suggested, pot_params_full = build_potential(PotentialConfig(key=pot_key, params=pot_params))

        c1, c2, c3 = st.columns(3)
        with c1:
            use_suggested_L = st.checkbox("Use suggested L", value=True, key="exc_useL")
        with c2:
            L = float(st.number_input("L", value=float(L_suggested), step=0.5, disabled=use_suggested_L, key="exc_L"))
        with c3:
            N = int(st.selectbox("N", options=[1024, 2048, 4096, 8192], index=2, key="exc_N"))

        if use_suggested_L:
            L = float(L_suggested)

        model_labels = available_models()
        model_key = st.selectbox("Architecture / ansatz", options=list(model_labels.keys()), format_func=lambda k: model_labels[k], key="exc_model")
        model_params = model_param_widgets(model_key, ui_key_prefix="exc_model")

        c1, c2, c3 = st.columns(3)
        with c1:
            K_states = int(st.slider("# states", min_value=2, max_value=8, value=4, step=1))
        with c2:
            compute_reference = st.checkbox("Compute FD reference", value=True, key="exc_ref")
        with c3:
            tag = st.text_input("run tag", value=f"excited_{pot_key}_{model_key}", key="exc_tag")

        st.markdown("#### Training")
        cfg = train_config_widgets(ui_key_prefix="exc_train", for_excited=True)

        dtype_choice = st.selectbox("dtype", options=["float64", "float32"], index=0, key="exc_dtype")

        if st.button("Run excited-state experiment", type="primary", key="exc_run"):
            dtype = torch.float64 if dtype_choice == "float64" else torch.float32
            torch.set_default_dtype(dtype)
            device = get_device()

            x_np, x, dx = make_grid(GridConfig(L=L, N=N), device=device, dtype=dtype)
            V_np = V_numpy(x_np)
            Vx = V_torch(x)

            ctx = RunContext.create(kind="excited", tag=tag)
            ctx.save_env()
            ctx.snapshot_repo()

            config = {
                "experiment_kind": "excited",
                "potential": {"key": pot_key, "params": pot_params_full},
                "grid": {"L": L, "N": N, "dx": dx},
                "model": {"key": model_key, "params": model_params},
                "train": cfg.__dict__,
                "K_states": K_states,
                "dtype": dtype_choice,
                "device": str(device),
            }
            ctx.save_config(config)

            st.pyplot(plot_potential(x_np, V_np, title="Potential"), clear_figure=True)

            evals_ref = None
            psis_ref = None
            if compute_reference:
                with st.spinner("Computing finite-difference reference (eigsh)..."):
                    evals_ref, psis_ref = fd_eigensolve_k(x_np, V_np, dx, k=max(6, K_states))
                    np.savez(ctx.data_dir / "reference_fd_k.npz", x=x_np, V=V_np, evals=evals_ref, psis=psis_ref)
                st.success("Computed FD reference.")

            model_cfg = ModelConfig(key=model_key, params=model_params)

            def build_model_fn(n: int) -> torch.nn.Module:
                # n=0 ground state: positive amplitude bias
                positive = (n == 0)
                return build_model(model_cfg, L=L, positive=positive).to(device)

            overall = st.progress(0)
            status = st.empty()

            def progress(state_name: str, n_state: int, step: int, row: Dict[str, float]):
                frac_state = (n_state + step / float(cfg.steps)) / float(K_states)
                overall.progress(int(100 * min(1.0, frac_state)))
                status.write({
                    "state": n_state,
                    "step": int(step),
                    "E": row["E"],
                    "loss": row["loss"],
                    "boundary_mass": row["boundary_mass"],
                })

            def checkpoint_fn(state_name: str, step: int, m: torch.nn.Module, meta: Dict[str, Any]):
                torch.save({"state": state_name, "step": step, "model_state": m.state_dict(), **meta}, ctx.ckpt_dir / f"{state_name}_step_{step:06d}.pt")

            with st.spinner("Training sequential states..."):
                states = train_states_sequential(
                    K_states=K_states,
                    build_model_fn=build_model_fn,
                    x=x,
                    Vx=Vx,
                    dx=dx,
                    cfg=cfg,
                    progress=progress,
                    checkpoint_fn=checkpoint_fn,
                )

            # Save per-state logs and arrays
            energies = []
            psis_nn = []
            for n, s in enumerate(states):
                energies.append(s["E"])
                psis_nn.append(s["psi"].detach().cpu().numpy())
                s["df"].to_csv(ctx.logs_dir / f"state_{n:02d}_train_log.csv", index=False)

            psis_nn = np.stack(psis_nn, axis=0)
            np.savez(ctx.data_dir / "variational_states.npz", x=x_np, V=V_np, E=np.array(energies), psis=psis_nn)

            # Visuals: training curves (state 0) + wavefunctions
            st.success("Training complete.")
            st.write(pd.DataFrame({"n": list(range(K_states)), "E": energies}))

            fig = plot_wavefunctions(x_np, psis_nn, labels=[f"nn psi{n}" for n in range(K_states)], title="Variational states")
            st.pyplot(fig, clear_figure=True)

            # Compare to reference
            if evals_ref is not None and psis_ref is not None:
                df_cmp, O = compare_states_to_reference(psis_nn, np.array(energies), psis_ref[:K_states], evals_ref[:K_states], dx)
                df_cmp.to_csv(ctx.data_dir / "comparison_to_reference.csv", index=False)
                st.subheader("Comparison to FD reference")
                st.dataframe(df_cmp, use_container_width=True)
                st.pyplot(plot_overlap_matrix(O, title="|Overlap| matrix"), clear_figure=True)

            # Save key figures
            save_fig(plot_potential(x_np, V_np, title="Potential"), ctx.figures_dir, "potential.png")
            save_fig(plot_wavefunctions(x_np, psis_nn, labels=[f"nn psi{n}" for n in range(K_states)], title="Variational states"), ctx.figures_dir, "nn_wavefunctions.png")

            # Download zip
            zip_path = ctx.zip_run()
            st.write("Run directory:", str(ctx.run_dir))
            with open(zip_path, "rb") as f:
                st.download_button("Download this run (zip)", data=f, file_name=zip_path.name, mime="application/zip")

    # -------------------- Hyperparameter sweep --------------------
    with tabs[2]:
        st.header("Hyperparameter sweep")

        pot_labels = available_potentials()
        pot_key = st.selectbox("Potential", options=list(pot_labels.keys()), format_func=lambda k: pot_labels[k], key="sw_pot")
        pot_params = potential_param_widgets(pot_key, ui_key_prefix="sw_pot")
        V_torch, V_numpy, L_suggested, pot_params_full = build_potential(PotentialConfig(key=pot_key, params=pot_params))

        c1, c2, c3 = st.columns(3)
        with c1:
            use_suggested_L = st.checkbox("Use suggested L", value=True, key="sw_useL")
        with c2:
            L = float(st.number_input("L", value=float(L_suggested), step=0.5, disabled=use_suggested_L, key="sw_L"))
        with c3:
            N = int(st.selectbox("N", options=[512, 1024, 2048, 4096], index=2, key="sw_N"))
        if use_suggested_L:
            L = float(L_suggested)

        model_labels = available_models()
        model_key = st.selectbox("Architecture / ansatz", options=list(model_labels.keys()), format_func=lambda k: model_labels[k], key="sw_model")

        c1, c2, c3 = st.columns(3)
        with c1:
            n_trials = int(st.slider("n_trials", min_value=3, max_value=60, value=12, step=1))
        with c2:
            steps = int(st.number_input("steps per trial", min_value=100, max_value=20000, value=1500, step=250))
        with c3:
            tag = st.text_input("run tag", value=f"sweep_{pot_key}_{model_key}", key="sw_tag")

        dtype_choice = st.selectbox("dtype", options=["float64", "float32"], index=0, key="sw_dtype")

        st.markdown("#### Sweep ranges")
        lr_choices = st.multiselect("lr choices", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], default=[3e-4, 1e-3, 3e-3])
        hidden_choices = st.multiselect("hidden choices", options=[32, 64, 128, 256], default=[64, 128])
        depth_choices = st.multiselect("depth choices", options=[2, 3, 4, 5], default=[3, 4])
        a_choices = st.multiselect("envelope a choices", options=[0.10, 0.25, 0.35, 0.60, 0.80], default=[0.25, 0.35, 0.60])

        if st.button("Run sweep", type="primary", key="sw_run"):
            dtype = torch.float64 if dtype_choice == "float64" else torch.float32
            torch.set_default_dtype(dtype)
            device = get_device()

            x_np, x, dx = make_grid(GridConfig(L=L, N=N), device=device, dtype=dtype)
            V_np = V_numpy(x_np)
            Vx = V_torch(x)

            ctx = RunContext.create(kind="sweep", tag=tag)
            ctx.save_env()
            ctx.snapshot_repo()

            config = {
                "experiment_kind": "sweep",
                "potential": {"key": pot_key, "params": pot_params_full},
                "grid": {"L": L, "N": N, "dx": dx},
                "model": {"key": model_key},
                "sweep": {
                    "n_trials": n_trials,
                    "steps": steps,
                    "lr_choices": lr_choices,
                    "hidden_choices": hidden_choices,
                    "depth_choices": depth_choices,
                    "a_choices": a_choices,
                },
                "dtype": dtype_choice,
                "device": str(device),
            }
            ctx.save_config(config)

            model_base_cfg = ModelConfig(key=model_key, params=None)

            rng = np.random.default_rng(1234)

            def sampler(t: int):
                lr = float(rng.choice(lr_choices))
                hidden = int(rng.choice(hidden_choices))
                depth = int(rng.choice(depth_choices))
                a = float(rng.choice(a_choices))

                # For simplicity, keep activation fixed unless model supports it
                model_params = {"hidden": hidden, "depth": depth, "a": a}
                if model_key in ["plain_mlp", "fourier_mlp"]:
                    model_params["act"] = "tanh"
                if model_key == "fourier_mlp":
                    model_params.update({"num_fourier": 24, "ff_scale": 2.5, "include_x": 1})
                if model_key == "siren":
                    model_params.update({"omega0": 30.0})
                if model_key == "rbf":
                    model_params.update({"M": 48, "init_span": 3.0, "init_sigma": 0.7})

                cfg = TrainConfig(steps=steps, lr=lr, weight_decay=1e-6, grad_clip=1.0, print_every=10_000, ui_update_every=max(10, steps // 10), ckpt_every=10_000, use_lbfgs=False, lam_ortho=0.0)
                return cfg, model_params

            def build_model_from_params(model_params: Dict[str, Any]):
                return build_model(ModelConfig(key=model_key, params=model_params), L=L, positive=True).to(device)

            prog_bar = st.progress(0)
            status = st.empty()

            def progress(trial: int, step: int, payload: Dict[str, Any]):
                frac = (trial + step / float(steps)) / float(n_trials)
                prog_bar.progress(int(100 * min(1.0, frac)))
                status.write({
                    "trial": int(trial),
                    "step": int(step),
                    "E": payload.get("E"),
                    "lr": payload.get("lr"),
                    "hidden": payload.get("hidden"),
                    "depth": payload.get("depth"),
                    "a": payload.get("a"),
                })

            with st.spinner("Running sweep..."):
                df = random_sweep(n_trials, sampler, build_model_from_params, x, Vx, dx, progress=progress)

            df.to_csv(ctx.data_dir / "sweep_results.csv", index=False)
            st.success("Sweep complete.")
            st.dataframe(df.sort_values("E").head(10), use_container_width=True)

            # Simple plots
            import plotly.express as px

            fig = px.scatter(df, x="lr", y="E", color="hidden", symbol="depth", title="E vs lr (color=hidden, symbol=depth)")
            st.plotly_chart(fig, use_container_width=True)

            fig = px.scatter(df, x="hidden", y="E", color="depth", title="E vs hidden")
            st.plotly_chart(fig, use_container_width=True)

            zip_path = ctx.zip_run()
            with open(zip_path, "rb") as f:
                st.download_button("Download this run (zip)", data=f, file_name=zip_path.name, mime="application/zip")

    # -------------------- Eigensolver --------------------
    with tabs[3]:
        st.header("Numerical eigensolver (finite difference)")

        pot_labels = available_potentials()
        pot_key = st.selectbox("Potential", options=list(pot_labels.keys()), format_func=lambda k: pot_labels[k], key="fd_pot")
        pot_params = potential_param_widgets(pot_key, ui_key_prefix="fd_pot")
        V_torch, V_numpy, L_suggested, pot_params_full = build_potential(PotentialConfig(key=pot_key, params=pot_params))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            use_suggested_L = st.checkbox("Use suggested L", value=True, key="fd_useL")
        with c2:
            L = float(st.number_input("L", value=float(L_suggested), step=0.5, disabled=use_suggested_L, key="fd_L"))
        with c3:
            N = int(st.selectbox("N", options=[1024, 2048, 4096, 8192, 16384], index=2, key="fd_N"))
        with c4:
            k = int(st.slider("k eigenstates", min_value=2, max_value=12, value=6, step=1))

        if use_suggested_L:
            L = float(L_suggested)

        tag = st.text_input("run tag", value=f"fd_{pot_key}", key="fd_tag")

        if st.button("Run eigensolver", type="primary", key="fd_run"):
            device = get_device()
            # FD solver uses numpy; dtype does not matter
            x_np = np.linspace(-L, L, N)
            dx = float(x_np[1] - x_np[0])
            V_np = V_numpy(x_np)

            ctx = RunContext.create(kind="fd", tag=tag)
            ctx.save_env()
            ctx.snapshot_repo()

            config = {
                "experiment_kind": "fd",
                "potential": {"key": pot_key, "params": pot_params_full},
                "grid": {"L": L, "N": N, "dx": dx},
                "k": k,
                "device": str(device),
            }
            ctx.save_config(config)

            st.pyplot(plot_potential(x_np, V_np, title="Potential"), clear_figure=True)

            with st.spinner("Running sparse eigensolver (eigsh)..."):
                evals, psis = fd_eigensolve_k(x_np, V_np, dx, k=k)

            np.savez(ctx.data_dir / "reference_fd_k.npz", x=x_np, V=V_np, evals=evals, psis=psis)
            st.success("Eigensolver complete.")

            st.write(pd.DataFrame({"n": list(range(k)), "E": evals}))
            st.pyplot(plot_wavefunctions(x_np, psis[: min(k, 6)], labels=[f"psi{n}" for n in range(min(k, 6))], title="First eigenfunctions"), clear_figure=True)

            save_fig(plot_potential(x_np, V_np, title="Potential"), ctx.figures_dir, "potential.png")
            save_fig(plot_wavefunctions(x_np, psis[: min(k, 6)], labels=[f"psi{n}" for n in range(min(k, 6))], title="First eigenfunctions"), ctx.figures_dir, "wavefunctions.png")

            zip_path = ctx.zip_run()
            with open(zip_path, "rb") as f:
                st.download_button("Download this run (zip)", data=f, file_name=zip_path.name, mime="application/zip")

    # -------------------- Trace browser --------------------
    with tabs[4]:
        st.header("Trace browser")
        st.caption("Inspect saved runs, configs, logs, and the code snapshot used to generate each run.")

        base = get_output_base_dir()
        runs_dir = base / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        run_paths = sorted([p for p in runs_dir.glob("*") if p.is_dir()], reverse=True)
        if not run_paths:
            st.info("No runs found yet. Run an experiment first.")
        else:
            selected = st.selectbox("Select a run", options=run_paths, format_func=lambda p: p.name)
            st.write("Run path:", str(selected))

            # Show config/env if available
            cfg_path = selected / "data" / "config.json"
            env_path = selected / "data" / "env.json"
            if cfg_path.exists():
                st.subheader("config.json")
                st.json(json.loads(cfg_path.read_text()))
            if env_path.exists():
                st.subheader("env.json")
                st.json(json.loads(env_path.read_text()))

            # Logs
            logs_dir = selected / "logs"
            if logs_dir.exists():
                log_files = sorted(logs_dir.glob("*.csv"))
                if log_files:
                    st.subheader("Logs")
                    lf = st.selectbox("Select log", options=log_files, format_func=lambda p: p.name)
                    df = pd.read_csv(lf)
                    st.dataframe(df.tail(200), use_container_width=True)

            # Code snapshot
            code_dir = selected / "code_snapshot"
            if code_dir.exists():
                st.subheader("Code snapshot")
                files = sorted([p for p in code_dir.rglob("*.py") if p.is_file()])
                if files:
                    cf = st.selectbox("Select a file", options=files, format_func=lambda p: str(p.relative_to(code_dir)))
                    st.code(cf.read_text(), language="python")

            # Download zip
            zip_path = selected.with_suffix(".zip")
            if zip_path.exists():
                with open(zip_path, "rb") as f:
                    st.download_button("Download run zip", data=f, file_name=zip_path.name, mime="application/zip")
            else:
                if st.button("Create zip for this run"):
                    ctx = RunContext(run_dir=selected)
                    z = ctx.zip_run()
                    st.success(f"Created: {z}")


if __name__ == "__main__":
    main()
