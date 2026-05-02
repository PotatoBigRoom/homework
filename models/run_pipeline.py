import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.build import fcc111
from matminer.datasets import load_dataset
from pymatgen.core import Composition, Element
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULT_DIR = ROOT / "results"
VIS_DIR = RESULT_DIR / "visualizations"
SEED = 42


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def ensure_dirs():
    for directory in (RAW_DIR, PROCESSED_DIR, RESULT_DIR, VIS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def clean_float(value, digits=4):
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if math.isnan(value) or math.isinf(value):
            return None
    return round(float(value), digits)


def composition_from_any(value):
    if isinstance(value, Composition):
        return value
    if isinstance(value, dict):
        return Composition(value)
    return Composition(str(value))


def weighted_stats(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    mean = float(np.sum(values * weights))
    variance = float(np.sum(weights * (values - mean) ** 2))
    return mean, float(np.sqrt(variance)), float(values.min()), float(values.max())


def composition_features(value):
    comp = composition_from_any(value)
    amounts = np.array([amt for _, amt in comp.items()], dtype=float)
    fractions = amounts / amounts.sum()
    elements = [el for el, _ in comp.items()]

    def prop(fn, fallback=0.0):
        output = []
        for el in elements:
            try:
                val = fn(el)
                val = float(val) if val is not None else fallback
                output.append(val if math.isfinite(val) else fallback)
            except Exception:
                output.append(fallback)
        return output

    z = prop(lambda el: el.Z)
    mass = prop(lambda el: el.atomic_mass)
    x = prop(lambda el: el.X, 0.0)
    row = prop(lambda el: el.row)
    group = prop(lambda el: el.group or 0)
    radius = prop(lambda el: el.atomic_radius or 0.0)

    features = []
    for values in (z, mass, x, row, group, radius):
        features.extend(weighted_stats(values, fractions))

    entropy = -float(np.sum(fractions * np.log(fractions + 1e-12)))
    features.extend(
        [
            float(len(elements)),
            float(amounts.sum()),
            float(fractions.max()),
            entropy,
            float(np.ptp(z)),
            float(np.ptp(x)),
        ]
    )
    return features


def structure_features(structure):
    comp_features = composition_features(structure.composition)
    volume_per_atom = structure.volume / max(len(structure), 1)
    density = structure.density
    return comp_features + [float(volume_per_atom), float(density), float(len(structure))]


def save_parity_plot(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=160)
    ax.scatter(y_true, y_pred, s=16, alpha=0.72, color="#1f7a5a", edgecolor="none")
    low = min(float(np.min(y_true)), float(np.min(y_pred)))
    high = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([low, high], [low, high], color="#a94b42", linewidth=1.4)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_battery_plot(df, path):
    fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=160)
    scatter = ax.scatter(
        df["pred_capacity"],
        df["pred_voltage"],
        c=df["pred_stability"],
        s=34,
        cmap="viridis",
        alpha=0.78,
        edgecolor="none",
    )
    ax.set_xlabel("Predicted capacity proxy")
    ax.set_ylabel("Predicted voltage proxy")
    ax.set_title("Battery candidate screening")
    ax.grid(alpha=0.22)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Predicted stability proxy")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_site_heatmap(df, path):
    pivot = df.pivot(index="metal", columns="site", values="pred_adsorption_energy")
    fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=160)
    image = ax.imshow(pivot.values, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title("Predicted adsorption energy by site")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("eV")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_alloy_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=160)
    image = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1], ["Not forming", "Forming"])
    ax.set_yticks([0, 1], ["Not forming", "Forming"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title("Alloy phase-forming confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontweight="bold")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


class FormationMLP(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, 96),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.08),
            torch.nn.Linear(96, 48),
            torch.nn.ReLU(),
            torch.nn.Linear(48, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_formation_model(flla):
    df = flla.sample(min(len(flla), 2600), random_state=SEED).reset_index(drop=True)
    X = np.array([structure_features(s) for s in df["structure"]], dtype=np.float32)
    y = df["formation_energy_per_atom"].astype(float).to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df.index.to_numpy(), test_size=0.2, random_state=SEED
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FormationMLP(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.SmoothL1Loss()

    x_tensor = torch.tensor(X_train, device=device)
    y_tensor = torch.tensor(y_train, device=device)
    for _ in range(180):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test, device=device)).cpu().numpy()

    plot_path = VIS_DIR / "crystal_stability_parity.png"
    save_parity_plot(y_test, pred, "Formation energy parity", plot_path)

    test_rows = df.loc[test_idx].copy()
    test_rows["predicted_formation_energy_per_atom"] = pred
    stable = test_rows.sort_values("predicted_formation_energy_per_atom").head(5)

    return {
        "id": "crystal-stability",
        "title": "晶体稳定性快速预测模型",
        "method": "PyTorch CUDA 11.8 MLP baseline with structure/composition features; CGCNN-ready data pipeline.",
        "dataset": "matminer flla, downloaded from Figshare; sampled 2600 structures.",
        "dataset_details": {
            "records_used": int(len(df)),
            "target": "formation_energy_per_atom",
            "split": "80/20 random train-test split, seed 42",
            "note": "This is a fast neural baseline using handcrafted structure/composition descriptors, not a full CGCNN benchmark.",
        },
        "visualization": "results/visualizations/crystal_stability_parity.png",
        "metrics": {
            "MAE": clean_float(mean_absolute_error(y_test, pred)),
            "R2": clean_float(r2_score(y_test, pred)),
            "Device": str(device),
        },
        "candidates": [
            {
                "material": row["material_id"],
                "formula": row["structure"].composition.reduced_formula,
                "predicted_e_form": clean_float(row["predicted_formation_energy_per_atom"]),
                "e_above_hull": clean_float(row["e_above_hull"]),
            }
            for _, row in stable.iterrows()
        ],
    }


def train_battery_model(oxides):
    df = oxides.copy()
    df = df.dropna(subset=["formula", "e_form", "e_hull", "gap pbe"]).reset_index(drop=True)
    df = df.sample(min(len(df), 2400), random_state=SEED).reset_index(drop=True)

    X = np.array([composition_features(x) for x in df["formula"]], dtype=float)
    formula_mass = np.array([composition_from_any(x).weight for x in df["formula"]], dtype=float)
    voltage_proxy = np.clip(-df["e_form"].astype(float).to_numpy() + 0.12 * df["gap pbe"].astype(float).to_numpy(), 0, 5)
    capacity_proxy = np.clip(26000 / formula_mass, 20, 320)
    stability_proxy = -df["e_hull"].astype(float).to_numpy()
    Y = np.vstack([voltage_proxy, capacity_proxy, stability_proxy]).T

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, Y, df.index.to_numpy(), test_size=0.22, random_state=SEED
    )
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=220, random_state=SEED, n_jobs=-1))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    eval_df = df.loc[test_idx].copy()
    eval_df[["pred_voltage", "pred_capacity", "pred_stability"]] = pred
    eval_df["score"] = (
        0.45 * eval_df["pred_voltage"]
        + 0.003 * eval_df["pred_capacity"]
        + 0.65 * eval_df["pred_stability"]
    )
    plot_path = VIS_DIR / "battery_screening_scatter.png"
    save_battery_plot(eval_df, plot_path)
    top = eval_df.sort_values("score", ascending=False).head(5)

    return {
        "id": "battery-cathode",
        "title": "锂离子电池正极材料筛选工具",
        "method": "Scikit-learn multi-output random forest with voltage/capacity/stability proxy targets.",
        "dataset": "matminer wolverton_oxides, downloaded from Figshare; oxide screening proxy when MP_API_KEY is absent.",
        "dataset_details": {
            "records_used": int(len(df)),
            "target": "proxy voltage, proxy capacity, proxy stability derived from e_form/e_hull/gap/formula mass",
            "split": "78/22 random train-test split, seed 42",
            "note": "This is not a real lithium intercalation voltage dataset. It is a runnable oxide-screening proxy because no Materials Project API key was provided.",
        },
        "visualization": "results/visualizations/battery_screening_scatter.png",
        "metrics": {
            "Voltage MAE": clean_float(mean_absolute_error(y_test[:, 0], pred[:, 0])),
            "Capacity MAE": clean_float(mean_absolute_error(y_test[:, 1], pred[:, 1])),
            "Stability R2": clean_float(r2_score(y_test[:, 2], pred[:, 2])),
        },
        "candidates": [
            {
                "material": row["formula"],
                "pred_voltage": clean_float(row["pred_voltage"]),
                "pred_capacity": clean_float(row["pred_capacity"]),
                "pred_stability": clean_float(row["pred_stability"]),
                "score": clean_float(row["score"]),
            }
            for _, row in top.iterrows()
        ],
    }


def train_catalysis_model():
    metals = ["Ni", "Cu", "Pd", "Ag", "Pt", "Au", "Co", "Rh", "Ir"]
    lattice_constants = {
        "Ni": 3.52,
        "Cu": 3.61,
        "Pd": 3.89,
        "Ag": 4.09,
        "Pt": 3.92,
        "Au": 4.08,
        "Co": 3.54,
        "Rh": 3.80,
        "Ir": 3.84,
    }
    sites = ["top", "bridge", "hollow"]
    rows = []
    for metal in metals:
        slab = fcc111(metal, size=(2, 2, 3), a=lattice_constants[metal], vacuum=8.0)
        element = Element(metal)
        for site_i, site in enumerate(sites):
            z = float(element.Z)
            x = float(element.X or 0)
            group = float(element.group or 0)
            coordination = 1.0 + site_i
            area = float(np.linalg.norm(np.cross(slab.cell[0], slab.cell[1])))
            adsorption_energy = -0.018 * z + 0.34 * x - 0.035 * group - 0.09 * coordination + 0.08 * np.sin(z)
            rows.append(
                {
                    "metal": metal,
                    "site": site,
                    "atomic_number": z,
                    "electronegativity": x,
                    "group": group,
                    "coordination": coordination,
                    "surface_area": area,
                    "slab_atoms": len(slab),
                    "adsorption_energy": adsorption_energy,
                }
            )

    df = pd.DataFrame(rows)
    X = df[["atomic_number", "electronegativity", "group", "coordination", "surface_area", "slab_atoms"]]
    y = df["adsorption_energy"]
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df.index.to_numpy(), test_size=0.3, random_state=SEED
    )
    model = GradientBoostingRegressor(random_state=SEED)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    scored = df.copy()
    scored["pred_adsorption_energy"] = model.predict(X)
    scored["activity_score"] = -abs(scored["pred_adsorption_energy"] + 0.8)
    plot_path = VIS_DIR / "catalyst_site_heatmap.png"
    save_site_heatmap(scored, plot_path)
    top = scored.sort_values("activity_score", ascending=False).head(5)

    return {
        "id": "catalyst-sites",
        "title": "催化活性位点智能识别系统",
        "method": "ASE slab generation plus gradient boosting adsorption-energy baseline.",
        "dataset": "ASE-generated fcc(111) surface/site dataset for Ni/Cu/Pd/Ag/Pt/Au/Co/Rh/Ir.",
        "dataset_details": {
            "records_used": int(len(df)),
            "target": "synthetic adsorption_energy from elemental/site descriptors",
            "split": "70/30 random train-test split, seed 42",
            "note": "R2 is high because the target is a deterministic synthetic formula over a small ASE-generated dataset. Replace with DFT adsorption energies for scientific use.",
        },
        "visualization": "results/visualizations/catalyst_site_heatmap.png",
        "metrics": {
            "Adsorption MAE": clean_float(mean_absolute_error(y_test, pred)),
            "Adsorption R2": clean_float(r2_score(y_test, pred)),
            "Surface Count": int(len(df)),
        },
        "candidates": [
            {
                "material": f"{row['metal']}(111)",
                "site": row["site"],
                "pred_adsorption_energy": clean_float(row["pred_adsorption_energy"]),
                "activity_score": clean_float(row["activity_score"]),
            }
            for _, row in top.iterrows()
        ],
    }


def train_alloy_model(glass):
    df = glass.copy().dropna(subset=["composition", "gfa"]).reset_index(drop=True)
    X = np.array([composition_features(x) for x in df["composition"]], dtype=float)
    y = df["gfa"].astype(bool).astype(int).to_numpy()

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df.index.to_numpy(), test_size=0.22, random_state=SEED, stratify=y
    )
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=260, random_state=SEED, class_weight="balanced", n_jobs=-1),
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, pred)
    plot_path = VIS_DIR / "alloy_confusion_matrix.png"
    save_alloy_confusion(cm, plot_path)

    eval_df = df.loc[test_idx].copy()
    eval_df["phase_stability_probability"] = proba
    top = eval_df.sort_values("phase_stability_probability", ascending=False).head(5)
    importances = model.named_steps["randomforestclassifier"].feature_importances_

    return {
        "id": "high-entropy-alloy",
        "title": "高熵合金相稳定性预测",
        "method": "Random Forest classifier with composition entropy and elemental statistics.",
        "dataset": "matminer matbench_glass, downloaded from Materials Project Matbench; used as metallic glass/phase-forming proxy.",
        "dataset_details": {
            "records_used": int(len(df)),
            "target": "gfa, glass forming ability",
            "split": "78/22 stratified train-test split, seed 42",
            "note": "This is a metallic-glass forming proxy, not a dedicated high-entropy alloy phase-diagram dataset.",
        },
        "visualization": "results/visualizations/alloy_confusion_matrix.png",
        "metrics": {
            "Accuracy": clean_float(accuracy_score(y_test, pred)),
            "F1": clean_float(f1_score(y_test, pred)),
            "Top Feature Weight": clean_float(importances.max()),
        },
        "candidates": [
            {
                "material": composition_from_any(row["composition"]).reduced_formula,
                "phase_probability": clean_float(row["phase_stability_probability"]),
                "observed_gfa": bool(row["gfa"]),
            }
            for _, row in top.iterrows()
        ],
    }


def main():
    ensure_dirs()
    flla = load_dataset("flla")
    oxides = load_dataset("wolverton_oxides")
    glass = load_dataset("matbench_glass")

    flla[["material_id", "formula", "e_above_hull", "formation_energy_per_atom"]].to_csv(
        RAW_DIR / "flla_subset.csv", index=False
    )
    oxides.to_csv(RAW_DIR / "wolverton_oxides.csv", index=False)
    glass.to_csv(RAW_DIR / "matbench_glass.csv", index=False)

    results = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "run_summary": {
            "command": "HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897 .venv/bin/python models/run_pipeline.py",
            "seed": SEED,
            "purpose": "Runnable homepage baseline and visualization pipeline. Several tasks use proxy datasets because no private Materials Project API key or DFT adsorption database was provided.",
        },
        "environment": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "data_sources": [
            "flla: Figshare via matminer",
            "wolverton_oxides: Figshare via matminer",
            "matbench_glass: Materials Project Matbench via matminer",
            "catalyst_sites: ASE-generated slab dataset",
        ],
        "models": [
            train_formation_model(flla),
            train_battery_model(oxides),
            train_catalysis_model(),
            train_alloy_model(glass),
        ],
    }

    output_path = RESULT_DIR / "model_results.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
