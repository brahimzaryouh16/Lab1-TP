# Partie 2 - classification multi-classes des pannes
# GridSearchCV + entrainement final + courbes + metriques + regularisation
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from data import download_data, load_and_clean, prepare_data, run_eda, RESULTS, FIGURES, METRICS, MODELS
from model import TorchMLPClassifier, build_mlp, make_optimizer

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(os.cpu_count() or 8)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward_logits(model, tensor, batch_size=8192):
    # passe avant par paquets pour eviter de saturer la memoire
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            outputs.append(model(tensor[i:i + batch_size]))
    return torch.cat(outputs)


def training_curves(config, max_epochs, patience=None, class_weights=False,
                    label_smoothing=0.0):
    """Entraine sur le train equilibre et suit loss/accuracy train et test a chaque epoch.
    Renvoie (modele, historique, nombre d'epochs effectuees)."""
    torch.manual_seed(SEED)

    X = torch.tensor(D["X_train_bal"], device=DEVICE)
    y = torch.tensor(D["y_train_bal"], device=DEVICE)
    X_test = torch.tensor(D["X_test"], device=DEVICE)
    y_test = torch.tensor(D["y_test"], device=DEVICE)

    model = build_mlp(X.shape[1], len(D["classes"]), tuple(config["hidden"]),
                      config.get("dropout", 0.0)).to(DEVICE)
    optimizer = make_optimizer(model, config.get("optimizer", "adam"),
                               config.get("lr", 0.001), config.get("weight_decay", 0.0))

    weight = None
    if class_weights:
        counts = np.bincount(np.asarray(D["y_train_bal"]))
        weight = torch.tensor(len(D["y_train_bal"]) / (len(D["classes"]) * counts),
                              dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    loader = DataLoader(TensorDataset(X, y),
                        batch_size=config.get("batch_size", 512), shuffle=True)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    best_test_loss = float("inf")
    best_state = None
    bad_epochs = 0
    epochs_ran = max_epochs

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        train_logits = forward_logits(model, X)
        test_logits = forward_logits(model, X_test)
        history["train_loss"].append(float(nn.functional.cross_entropy(train_logits, y)))
        history["test_loss"].append(float(nn.functional.cross_entropy(test_logits, y_test)))
        history["train_acc"].append(float((train_logits.argmax(1) == y).float().mean()))
        history["test_acc"].append(float((test_logits.argmax(1) == y_test).float().mean()))

        if patience:
            test_loss = history["test_loss"][-1]
            if test_loss < best_test_loss - 1e-5:
                best_test_loss = test_loss
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


def predict_classes(model, X):
    # accepte soit un classifieur sklearn (predict), soit un modele PyTorch brut
    if hasattr(model, "predict"):
        return model.predict(X)
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            xb = torch.tensor(X[i:i + 8192], device=DEVICE)
            predictions.append(model(xb).argmax(dim=1))
    return torch.cat(predictions).cpu().numpy()


def evaluate(model, X, y):
    predictions = predict_classes(model, X)
    metrics = {
        "accuracy": round(float(accuracy_score(y, predictions)), 4),
        "sensitivity_recall_macro": round(float(recall_score(y, predictions, average="macro", zero_division=0)), 4),
        "sensitivity_recall_weighted": round(float(recall_score(y, predictions, average="weighted", zero_division=0)), 4),
        "precision_macro": round(float(precision_score(y, predictions, average="macro", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y, predictions, average="weighted", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y, predictions, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y, predictions, average="weighted", zero_division=0)), 4),
    }
    return metrics, predictions


# ---------------------------------------------------------------------------
# 1. Donnees : pretraitement, EDA et equilibrage
# ---------------------------------------------------------------------------
download_data()
df, preprocessing_report = load_and_clean()
D = prepare_data(df)
run_eda(df, D["y_all"], D["classes"], D["counts_before"], D["counts_after"])
print("Train=%d | Test=%d | Train equilibre (SMOTE)=%d"
      % (D["n_train"], D["n_test"], D["n_train_bal"]))

# ---------------------------------------------------------------------------
# 2. GridSearchCV (sklearn) pour choisir les hyper-parametres
# ---------------------------------------------------------------------------
param_grid = {
    "hidden": [(64, 32), (128, 64), (128, 64, 32)],
    "lr": [0.01, 0.001],
    "optimizer": ["adam", "sgd"],
}
grid = GridSearchCV(
    TorchMLPClassifier(epochs=30, batch_size=512),
    param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
    scoring="f1_macro",  # l'accuracy seule est trompeuse (classes desequilibrees)
    n_jobs=1,
    verbose=1,
)
grid.fit(D["X_train_bal"], D["y_train_bal"])

best_params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in grid.best_params_.items()}
print("Meilleurs parametres:", best_params, "| F1-macro cv = %.4f" % grid.best_score_)

cv_results = pd.DataFrame(grid.cv_results_)[
    ["param_hidden", "param_lr", "param_optimizer",
     "mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time"]]
cv_results = cv_results.sort_values("rank_test_score")
cv_results.to_csv(os.path.join(METRICS, "grid_results.csv"), index=False)
with open(os.path.join(METRICS, "best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)

# ---------------------------------------------------------------------------
# 3. Entrainement final + courbes
# ---------------------------------------------------------------------------
print("Entrainement final...")
best = dict(best_params)
best["batch_size"] = 512
best["epochs"] = 30
model, history, epochs_ran = training_curves(best, max_epochs=best["epochs"])

epochs = range(1, len(history["train_loss"]) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, history["train_loss"], "o-", label="Train")
plt.plot(epochs, history["test_loss"], "s-", label="Test")
plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss")
plt.title("Loss / Epochs - classification des pannes")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "p2_05_curves_loss.png"), dpi=110)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs, history["train_acc"], "o-", label="Train")
plt.plot(epochs, history["test_acc"], "s-", label="Test")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy / Epochs - classification des pannes")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "p2_06_curves_accuracy.png"), dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# 4. Metriques sur train (equilibre) et test (distribution reelle)
# ---------------------------------------------------------------------------
train_metrics, train_predictions = evaluate(model, D["X_train_bal"], D["y_train_bal"])
test_metrics, test_predictions = evaluate(model, D["X_test"], D["y_test"])
with open(os.path.join(METRICS, "final_metrics.json"), "w") as f:
    json.dump({"train_balanced": train_metrics, "test_original": test_metrics}, f, indent=2)
print("Metriques train (equilibre):", train_metrics)
print("Metriques test (original)  :", test_metrics)

report_train = classification_report(D["y_train_bal"], train_predictions,
                                     target_names=D["classes"], digits=3, zero_division=0)
report_test = classification_report(D["y_test"], test_predictions,
                                    target_names=D["classes"], digits=3, zero_division=0)
with open(os.path.join(METRICS, "classification_reports.txt"), "w") as f:
    f.write("=== TRAIN (equilibre) ===\n" + report_train
            + "\n\n=== TEST (distribution originale) ===\n" + report_test)

plt.figure(figsize=(13, 5.2))
cm = confusion_matrix(D["y_test"], test_predictions)
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=D["classes"], yticklabels=D["classes"])
plt.title("Matrice de confusion - Test (compte)")
plt.ylabel("Reel")
plt.xlabel("Predit")
plt.subplot(1, 2, 2)
sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=True, fmt=".2f", cmap="Blues",
            xticklabels=D["classes"], yticklabels=D["classes"])
plt.title("Matrice de confusion - Test (normalisee)")
plt.ylabel("Reel")
plt.xlabel("Predit")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "p2_07_confusion_matrix.png"), dpi=110)
plt.close()

f1_per_class = f1_score(D["y_test"], test_predictions, average=None,
                        labels=np.arange(len(D["classes"])), zero_division=0)
plt.figure(figsize=(9, 4.5))
plt.bar(D["classes"], f1_per_class)
plt.ylabel("F1-score")
plt.title("F1-score par classe (test)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "p2_08_f1_per_class.png"), dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# 5. Comparaison des techniques de regularisation
# ---------------------------------------------------------------------------
print("Comparaison des techniques de regularisation...")
base_config = {k: v for k, v in best.items() if k != "epochs"}
variants = [
    ("Baseline (SMOTE)", dict(base_config), {}),
    ("+ Dropout 0.3", dict(base_config, dropout=0.3), {}),
    ("+ L2 (weight decay 1e-4)", dict(base_config, weight_decay=1e-4), {}),
    ("+ Dropout + L2", dict(base_config, dropout=0.3, weight_decay=1e-4), {}),
    ("+ Poids de classes", dict(base_config), {"class_weights": True}),
    ("+ Early stopping (p=6)", dict(base_config), {"patience": 6}),
    ("+ Label smoothing 0.05", dict(base_config), {"label_smoothing": 0.05}),
]

rows, results = [], {}
for name, config, extra in variants:
    variant_model, _, ran = training_curves(config, max_epochs=best["epochs"], **extra)
    variant_train_metrics, _ = evaluate(variant_model, D["X_train_bal"], D["y_train_bal"])
    variant_test_metrics, _ = evaluate(variant_model, D["X_test"], D["y_test"])
    rows.append({"Variante": name, "Epochs_effectives": ran,
                 "Test_Accuracy": variant_test_metrics["accuracy"],
                 "Test_F1_macro": variant_test_metrics["f1_macro"],
                 "Test_Recall_macro": variant_test_metrics["sensitivity_recall_macro"],
                 "Train_F1_macro": variant_train_metrics["f1_macro"],
                 "Gap_F1(train-test)": round(variant_train_metrics["f1_macro"]
                                             - variant_test_metrics["f1_macro"], 4)})
    results[name] = {"config": config, "train_balanced": variant_train_metrics,
                     "test_original": variant_test_metrics}
    print("  %-28s -> Test Acc=%.4f | Test F1-macro=%.4f"
          % (name, variant_test_metrics["accuracy"], variant_test_metrics["f1_macro"]))
    if name == "Baseline (SMOTE)":
        torch.save(variant_model.state_dict(), os.path.join(MODELS, "model_baseline.pt"))

# modele entraine sans augmentation, pour montrer l'apport de SMOTE
print("  Comparaison: sans augmentation ...")
no_augmentation = TorchMLPClassifier(hidden=tuple(best["hidden"]), lr=best["lr"],
                                     optimizer=best["optimizer"], epochs=30, batch_size=512)
no_augmentation.fit(D["X_train"], D["y_train"])
noaug_train_metrics, _ = evaluate(no_augmentation, D["X_train"], D["y_train"])
noaug_test_metrics, _ = evaluate(no_augmentation, D["X_test"], D["y_test"])
rows.insert(1, {"Variante": "Sans augmentation", "Epochs_effectives": 30,
                "Test_Accuracy": noaug_test_metrics["accuracy"],
                "Test_F1_macro": noaug_test_metrics["f1_macro"],
                "Test_Recall_macro": noaug_test_metrics["sensitivity_recall_macro"],
                "Train_F1_macro": noaug_train_metrics["f1_macro"],
                "Gap_F1(train-test)": round(noaug_train_metrics["f1_macro"]
                                            - noaug_test_metrics["f1_macro"], 4)})
results["Sans augmentation"] = {"config": base_config, "train_balanced": noaug_train_metrics,
                                "test_original": noaug_test_metrics}
print("  %-28s -> Test Acc=%.4f | Test F1-macro=%.4f"
      % ("Sans augmentation", noaug_test_metrics["accuracy"], noaug_test_metrics["f1_macro"]))

comparison = pd.DataFrame(rows)
comparison.to_csv(os.path.join(METRICS, "regularization_comparison.csv"), index=False)
with open(os.path.join(METRICS, "regularization_results.json"), "w") as f:
    json.dump(results, f, indent=2)

short_names = [r["Variante"].replace("+ ", "") for r in rows]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
ax[0].bar(short_names, [r["Test_Accuracy"] for r in rows])
ax[0].set_title("Accuracy test")
ax[0].set_ylim(0.8, 1.005)
ax[1].bar(short_names, [r["Test_F1_macro"] for r in rows])
ax[1].set_title("F1-macro test")
ax[2].bar(short_names, [r["Gap_F1(train-test)"] for r in rows])
ax[2].set_title("Ecart F1 train - F1 test")
for a in ax:
    a.tick_params(axis="x", rotation=35)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "p2_09_regularization.png"), dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# 6. Sauvegarde du modele et des infos
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), os.path.join(MODELS, "model_final.pt"))
with open(os.path.join(METRICS, "run_info.json"), "w") as f:
    json.dump({"best_params": best_params, "classes": D["classes"],
               "feature_names": D["feature_names"],
               "counts_before_smote": {D["classes"][i]: int(D["counts_before"].get(i, 0))
                                       for i in range(len(D["classes"]))},
               "counts_after_smote": {D["classes"][i]: int(D["counts_after"].get(i, 0))
                                      for i in range(len(D["classes"]))},
               "n_train_original": D["n_train"], "n_test": D["n_test"],
               "n_train_balanced": D["n_train_bal"]}, f, indent=2)

with open(os.path.join(METRICS, "preprocessing_report.txt"), "w") as f:
    f.write("\n".join(preprocessing_report))

print("Termine. Resultats dans", RESULTS)
print(comparison.to_string(index=False))
