# Partie 1 - regression sur le dataset NYSE
# GridSearchCV + entrainement final + courbes + regularisation
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from data import download_data, load_data, run_eda, prepare_data, RESULTS
from model import TorchMLPRegressor, build_mlp, make_optimizer

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(os.cpu_count() or 8)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tolerance_accuracy(y_true, y_pred, tolerance):
    # pourcentage de predictions dans la marge de +-tolerance
    return float(np.mean(np.abs((y_pred - y_true) / y_true) < tolerance))


def full_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": round(float(mse), 4),
        "RMSE": round(float(np.sqrt(mse)), 4),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "MAPE_%": round(float(np.mean(np.abs((y_pred - y_true) / y_true)) * 100), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 6),
        "Acc@1%": round(tolerance_accuracy(y_true, y_pred, 0.01), 4),
        "Acc@2%": round(tolerance_accuracy(y_true, y_pred, 0.02), 4),
        "Acc@5%": round(tolerance_accuracy(y_true, y_pred, 0.05), 4),
    }


def predict_batches(model, X, batch_size=8192):
    # prediction par paquets pour ne pas saturer la memoire
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], device=DEVICE)
            outputs.append(model(xb).view(-1))
    return torch.cat(outputs).cpu().numpy()


def train_model(config, max_epochs, early_stopping=False, patience=5, val_fraction=0.1):
    """Entraine un MLP et enregistre les metriques train/test a chaque epoch.
    Renvoie (modele, historique, nombre d'epochs effectuees)."""
    torch.manual_seed(SEED)

    if early_stopping:
        # les 10% les plus recents du train servent a surveiller l'entrainement
        n_val = int(len(D["X_train"]) * val_fraction)
    else:
        n_val = 0
    X_fit = D["X_train"][:-n_val] if n_val else D["X_train"]
    y_fit = D["y_train_std"][:-n_val] if n_val else D["y_train_std"]
    X_val = D["X_train"][-n_val:] if n_val else None
    y_val = D["y_train_std"][-n_val:] if n_val else None

    model = build_mlp(X_fit.shape[1], tuple(config["hidden"]), config.get("dropout", 0.0)).to(DEVICE)
    optimizer = make_optimizer(model, config.get("optimizer", "adam"),
                               config.get("lr", 0.001), config.get("weight_decay", 0.0))
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_fit, device=DEVICE),
                      torch.tensor(y_fit, device=DEVICE).view(-1, 1)),
        batch_size=config.get("batch_size", 1024), shuffle=True)

    history = {"train_loss": [], "test_loss": [], "train_r2": [], "test_r2": [],
               "train_acc2": [], "test_acc2": []}
    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0
    epochs_ran = max_epochs

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            if config.get("l1", 0.0) > 0:
                loss = loss + config["l1"] * sum(p.abs().sum() for p in model.parameters())
            loss.backward()
            optimizer.step()

        # loss (espace normalise) et metriques en dollars sur train et test
        train_pred_std = predict_batches(model, D["X_train"])
        test_pred_std = predict_batches(model, D["X_test"])
        history["train_loss"].append(float(np.mean((train_pred_std - D["y_train_std"]) ** 2)))
        history["test_loss"].append(float(np.mean((test_pred_std - D["y_test_std"]) ** 2)))

        train_pred = D["scaler_y"].inverse_transform(train_pred_std.reshape(-1, 1)).ravel()
        test_pred = D["scaler_y"].inverse_transform(test_pred_std.reshape(-1, 1)).ravel()
        history["train_r2"].append(float(r2_score(D["y_train"], train_pred)))
        history["test_r2"].append(float(r2_score(D["y_test"], test_pred)))
        history["train_acc2"].append(tolerance_accuracy(D["y_train"], train_pred, 0.02))
        history["test_acc2"].append(tolerance_accuracy(D["y_test"], test_pred, 0.02))

        if n_val:
            val_pred = predict_batches(model, X_val)
            val_loss = float(np.mean((val_pred - y_val) ** 2))
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                bad_epochs = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    epochs_ran = epoch + 1
                    print("  Early stopping a l'epoch", epochs_ran)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, epochs_ran


# ---------------------------------------------------------------------------
# 1. Donnees + EDA
# ---------------------------------------------------------------------------
download_data()
prices, securities = load_data()
run_eda(prices, securities)

D = prepare_data(prices)
print("Samples: train=%d (jusqu'a %s) | test=%d | features=%d"
      % (D["n_train"], D["split_date"], D["n_test"], D["X_train"].shape[1]))

# ---------------------------------------------------------------------------
# 2. GridSearchCV (sklearn) pour choisir les hyper-parametres
# ---------------------------------------------------------------------------
# recherche sur un sous-echantillon pour aller plus vite (on garde l'ordre du temps)
step = max(1, D["n_train"] // 120000)
X_search = D["X_train"][::step]
y_search = D["y_train_std"][::step]
print("Grid search sur %d points" % len(X_search))

param_grid = {
    "hidden": [(64, 32), (128, 64, 32), (128, 64)],
    "lr": [0.01, 0.001],
    "optimizer": ["adam", "sgd"],
    "epochs": [15, 40],
}
grid = GridSearchCV(
    TorchMLPRegressor(batch_size=1024),
    param_grid,
    cv=TimeSeriesSplit(n_splits=2),  # pas de K-fold : on respecte le temps
    scoring="neg_mean_squared_error",
    n_jobs=1,
    verbose=1,
)
grid.fit(X_search, y_search)

best_params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in grid.best_params_.items()}
best_params["batch_size"] = 1024
print("Meilleurs parametres:", best_params, "| MSE cv = %.5f" % -grid.best_score_)

cv_results = pd.DataFrame(grid.cv_results_)[
    ["param_hidden", "param_lr", "param_optimizer", "param_epochs",
     "mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time"]]
cv_results["mean_test_score"] = -cv_results["mean_test_score"]
cv_results = cv_results.sort_values("rank_test_score")
cv_results.to_csv(os.path.join(RESULTS, "grid_results.csv"), index=False)
with open(os.path.join(RESULTS, "best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)

# ---------------------------------------------------------------------------
# 3. Entrainement final + courbes
# ---------------------------------------------------------------------------
print("Entrainement final...")
best = dict(best_params)
model, history, final_epochs = train_model(best, max_epochs=best["epochs"])

train_pred = D["scaler_y"].inverse_transform(
    predict_batches(model, D["X_train"]).reshape(-1, 1)).ravel()
test_pred = D["scaler_y"].inverse_transform(
    predict_batches(model, D["X_test"]).reshape(-1, 1)).ravel()
metrics = {"train": full_metrics(D["y_train"], train_pred),
           "test": full_metrics(D["y_test"], test_pred)}
with open(os.path.join(RESULTS, "final_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Metriques finales train:", metrics["train"])
print("Metriques finales test :", metrics["test"])

epochs = range(1, len(history["train_loss"]) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, history["train_loss"], "o-", label="Train")
plt.plot(epochs, history["test_loss"], "s-", label="Test")
plt.xlabel("Epochs")
plt.ylabel("MSE (y normalise)")
plt.title("Loss / Epochs - modele final (regression NYSE)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "p1_05_curves_loss.png"), dpi=110)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs, history["train_r2"], "o-", label="R2 Train")
plt.plot(epochs, history["test_r2"], "s-", label="R2 Test")
plt.plot(epochs, history["train_acc2"], "--", label="Accuracy@2% Train")
plt.plot(epochs, history["test_acc2"], "--", label="Accuracy@2% Test")
plt.xlabel("Epochs")
plt.ylabel("R2 / Accuracy")
plt.title("Accuracy / Epochs - modele final (R2 et tolerance 2%)")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "p1_06_curves_accuracy.png"), dpi=110)
plt.close()

with open(os.path.join(RESULTS, "history_final.json"), "w") as f:
    json.dump(history, f)

# nuage de points predictions / vraies valeurs + residus
rng = np.random.default_rng(0)
idx = rng.choice(len(D["y_test"]), size=min(20000, len(D["y_test"])), replace=False)
plt.figure(figsize=(11, 4.5))
plt.subplot(1, 2, 1)
plt.scatter(D["y_test"][idx], test_pred[idx], s=3, alpha=0.25)
lim = [min(D["y_test"].min(), test_pred.min()), max(D["y_test"].max(), test_pred.max())]
plt.plot(lim, lim, "r--", linewidth=1)
plt.xlabel("Prix reel ($)")
plt.ylabel("Prix predit ($)")
plt.title("Test : prediction vs realite")
plt.subplot(1, 2, 2)
plt.hist(test_pred[idx] - D["y_test"][idx], bins=100, alpha=0.85)
plt.xlabel("Erreur de prediction ($)")
plt.ylabel("Nb d'echantillons")
plt.title("Distribution des residus (test)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "p1_07_predictions.png"), dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# 4. Comparaison des techniques de regularisation
# ---------------------------------------------------------------------------
print("Comparaison des techniques de regularisation...")
base_config = {k: v for k, v in best.items() if k != "epochs"}
variants = [
    ("Baseline", dict(base_config)),
    ("+ Dropout 0.2", dict(base_config, dropout=0.2)),
    ("+ L2 (weight decay 1e-4)", dict(base_config, weight_decay=1e-4)),
    ("+ Dropout 0.2 + L2", dict(base_config, dropout=0.2, weight_decay=1e-4)),
    ("+ L1 (penalite 1e-5)", dict(base_config, l1=1e-5)),
    ("+ Early stopping (p=5)", dict(base_config)),
]

rows, results = [], {}
for name, config in variants:
    early_stopping = "Early" in name
    variant_model, _, ran = train_model(config, max_epochs=best["epochs"],
                                        early_stopping=early_stopping)
    variant_pred = D["scaler_y"].inverse_transform(
        predict_batches(variant_model, D["X_test"]).reshape(-1, 1)).ravel()
    variant_train_pred = D["scaler_y"].inverse_transform(
        predict_batches(variant_model, D["X_train"]).reshape(-1, 1)).ravel()
    train_metrics = full_metrics(D["y_train"], variant_train_pred)
    test_metrics = full_metrics(D["y_test"], variant_pred)
    rows.append({"Variante": name, "Epochs_effectives": ran,
                 "Train_RMSE": train_metrics["RMSE"], "Test_RMSE": test_metrics["RMSE"],
                 "Train_R2": train_metrics["R2"], "Test_R2": test_metrics["R2"],
                 "Test_MAPE_%": test_metrics["MAPE_%"], "Test_Acc@2%": test_metrics["Acc@2%"],
                 "Gap_R2(train-test)": round(train_metrics["R2"] - test_metrics["R2"], 6)})
    results[name] = {"config": config, "train": train_metrics, "test": test_metrics}
    print("  %-28s -> Test RMSE=%.4f | Test R2=%.5f" % (name, test_metrics["RMSE"], test_metrics["R2"]))
    if name == "Baseline":
        torch.save(variant_model.state_dict(), os.path.join(RESULTS, "model_baseline.pt"))

comparison = pd.DataFrame(rows)
comparison.to_csv(os.path.join(RESULTS, "regularization_comparison.csv"), index=False)
with open(os.path.join(RESULTS, "regularization_results.json"), "w") as f:
    json.dump(results, f, indent=2)

short_names = [r["Variante"].replace("+ ", "").replace(" (p=5)", "") for r in rows]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
ax[0].bar(short_names, [r["Test_RMSE"] for r in rows])
ax[0].set_title("RMSE test")
ax[1].bar(short_names, [r["Test_R2"] for r in rows])
ax[1].set_title("R2 test")
ax[2].bar(short_names, [r["Gap_R2(train-test)"] for r in rows])
ax[2].set_title("Ecart de generalisation (R2 train - R2 test)")
for a in ax:
    a.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "p1_08_regularization.png"), dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# 5. Sauvegarde du modele et des infos
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), os.path.join(RESULTS, "model_final.pt"))
with open(os.path.join(RESULTS, "run_info.json"), "w") as f:
    json.dump({"best_params": best_params, "final_config": best,
               "epochs_effectives_final": final_epochs,
               "n_train": D["n_train"], "n_test": D["n_test"],
               "features": D["feature_names"], "L": 5,
               "split_date": str(D["split_date"])}, f, indent=2)

print("Termine. Resultats dans", RESULTS)
print(comparison.to_string(index=False))
