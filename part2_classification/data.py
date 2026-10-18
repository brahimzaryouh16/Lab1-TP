# Partie 2 - donnees : telechargement, pretraitement, EDA et equilibrage (SMOTE)
import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

CSV_URL = ("https://raw.githubusercontent.com/Pharezzzz/"
           "Predictive-Maintenance-Pipeline-with-Machine-Learning/main/predictive_maintenance.csv")
CSV_PATH = os.path.join(DATA_DIR, "predictive_maintenance.csv")

SEED = 42

NUM_COLS = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
            "Torque [Nm]", "Tool wear [min]", "Power [W]", "Temp diff [K]"]


def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        print("Telechargement:", CSV_URL)
        urllib.request.urlretrieve(CSV_URL, CSV_PATH)


def load_and_clean():
    raw = pd.read_csv(CSV_PATH)

    report = ["Shape brut: %s" % str(raw.shape),
              "Valeurs manquantes: %d" % raw.isnull().sum().sum(),
              "Doublons: %d" % raw.duplicated().sum(), "",
              raw.describe().to_string(), ""]

    df = raw.copy()
    df["Product_Type"] = df["Product ID"].str[0]
    # on enleve les identifiants et la colonne binaire Target (redondante)
    df = df.drop(columns=["UDI", "Product ID", "Type", "Target"])

    # deux variables calculees a partir de la physique de la machine
    df["Power [W]"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60
    df["Temp diff [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    report += ["Apres pretraitement: %s" % str(df.shape),
               "Colonnes: " + ", ".join(df.columns), ""]
    return df, report


def prepare_data(df):
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(df["Failure Type"])
    classes = list(label_encoder.classes_)

    X_num = df[NUM_COLS].to_numpy(dtype=np.float32)
    dummies = pd.get_dummies(df["Product_Type"], prefix="Type").astype(np.float32)
    X_all = np.hstack([X_num, dummies.to_numpy(dtype=np.float32)])
    feature_names = NUM_COLS + list(dummies.columns)

    # split stratifie AVANT equilibrage : le test garde la distribution reelle
    idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.2,
                                           stratify=y_all, random_state=SEED)
    X_train_raw, X_test_raw = X_all[idx_train], X_all[idx_test]
    y_train, y_test = y_all[idx_train], y_all[idx_test]

    # standardisation (calculee sur le train uniquement)
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    counts_before = pd.Series(y_train).value_counts().sort_index()

    # equilibrage du train avec SMOTE (le test n'est jamais touche)
    smote = SMOTE(k_neighbors=3, random_state=SEED)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    counts_after = pd.Series(y_train_bal).value_counts().sort_index()

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test, "y_all": y_all,
        "X_train_bal": X_train_bal, "y_train_bal": y_train_bal,
        "classes": classes, "feature_names": feature_names,
        "counts_before": counts_before, "counts_after": counts_after,
        "n_train": len(X_train), "n_test": len(X_test),
        "n_train_bal": len(X_train_bal),
    }


def run_eda(df, y_all, classes, counts_before, counts_after):
    n_classes = len(classes)

    # figure 1 : distribution des classes + effet de SMOTE
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    counts_all = pd.Series(y_all).value_counts().sort_index()
    plt.bar(classes, counts_all.values)
    plt.title("Distribution des classes (dataset complet)")
    plt.xticks(rotation=25, ha="right")
    plt.subplot(1, 2, 2)
    positions = np.arange(n_classes)
    plt.bar(positions - 0.2, [counts_before.get(i, 0) for i in range(n_classes)],
            width=0.4, label="Avant SMOTE")
    plt.bar(positions + 0.2, [counts_after.get(i, 0) for i in range(n_classes)],
            width=0.4, label="Apres SMOTE")
    plt.yscale("log")
    plt.xticks(positions, classes, rotation=25, ha="right")
    plt.title("Equilibrage du train (SMOTE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "p2_01_class_balance.png"), dpi=110)
    plt.close()

    # figure 2 : distribution des variables capteurs par classe
    plt.figure(figsize=(14, 4.2))
    for k, col in enumerate(["Torque [Nm]", "Rotational speed [rpm]", "Tool wear [min]"]):
        plt.subplot(1, 3, k + 1)
        for c in range(n_classes):
            mask = y_all == c
            plt.hist(df[col][mask], bins=40, alpha=0.45, density=True, label=classes[c])
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("Densite")
    plt.legend(fontsize=7)
    plt.suptitle("Distribution des variables capteurs par type de panne")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "p2_02_features_by_class.png"), dpi=110)
    plt.close()

    # figure 3 : correlations (variables numeriques + id de classe)
    corr_df = df[NUM_COLS].copy()
    corr_df["failure_class_id"] = y_all
    plt.figure(figsize=(7.5, 6))
    sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                annot_kws={"size": 7})
    plt.title("Correlations (features numeriques + id de classe)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "p2_03_correlation.png"), dpi=110)
    plt.close()

    # figure 4 : vitesse x couple colore par type de panne
    plt.figure(figsize=(8, 5.5))
    for c in range(n_classes):
        mask = y_all == c
        plt.scatter(df["Rotational speed [rpm]"][mask], df["Torque [Nm]"][mask],
                    s=8, alpha=0.5, label=classes[c])
    plt.xlabel("Vitesse de rotation (rpm)")
    plt.ylabel("Couple (Nm)")
    plt.title("Vitesse x Couple colore par type de panne (loi puissance)")
    plt.legend(fontsize=8, markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "p2_04_scatter_rpm_torque.png"), dpi=110)
    plt.close()
