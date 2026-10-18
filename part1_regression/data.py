# Partie 1 - donnees : telechargement, EDA et construction des features
import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(RESULTS, "figures")   # courbes et figures
METRICS = os.path.join(RESULTS, "metrics")   # json, csv et rapports texte
MODELS = os.path.join(RESULTS, "models")     # poids des modeles (.pt)
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(METRICS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

PRICES_URL = "https://raw.githubusercontent.com/kyi3081/stock-analysis/master/prices-split-adjusted.csv"
SECURITIES_URL = "https://raw.githubusercontent.com/kyi3081/stock-analysis/master/securities.csv"
PRICES_CSV = os.path.join(DATA_DIR, "prices-split-adjusted.csv")
SECURITIES_CSV = os.path.join(DATA_DIR, "securities.csv")

L = 5  # nombre de jours utilises en entree (une semaine de trading)
FEATURES = ["open", "close", "low", "high", "volume"]


def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    for path, url in [(PRICES_CSV, PRICES_URL), (SECURITIES_CSV, SECURITIES_URL)]:
        if not os.path.exists(path):
            print("Telechargement:", url)
            urllib.request.urlretrieve(url, path)


def load_data():
    prices = pd.read_csv(PRICES_CSV, parse_dates=["date"])
    securities = pd.read_csv(SECURITIES_CSV)
    securities.columns = [c.replace('"', "") for c in securities.columns]
    return prices, securities


def run_eda(prices, securities):
    # petit resume texte du dataset
    with open(os.path.join(METRICS, "eda_summary.txt"), "w") as f:
        f.write("Shape: %s\n" % str(prices.shape))
        f.write("Periode: %s -> %s\n" % (prices["date"].min().date(), prices["date"].max().date()))
        f.write("Nombre de symboles: %d\n" % prices["symbol"].nunique())
        f.write("Valeurs manquantes: %d\n" % prices.isnull().sum().sum())
        f.write("Doublons: %d\n\n" % prices.duplicated().sum())
        f.write(prices.describe().to_string())

    # figure 1 : evolution du prix de quelques actions connues
    known = ["AAPL", "MSFT", "AMZN", "GOOGL", "XOM", "JPM"]
    symbols = set(prices["symbol"].unique())
    known = [s for s in known if s in symbols]
    plt.figure(figsize=(11, 5))
    for sym in known:
        sub = prices[prices["symbol"] == sym]
        plt.plot(sub["date"], sub["close"], linewidth=1, label=sym)
    plt.title("Prix de cloture ajuste - quelques valeurs connues")
    plt.xlabel("Date")
    plt.ylabel("Prix ($)")
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "p1_01_price_history.png"), dpi=110)
    plt.close()

    # figure 2 : distribution du prix et du volume
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.hist(prices["close"], bins=120, alpha=0.8)
    plt.title("Distribution du prix de cloture")
    plt.xlabel("Prix ($)")
    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(prices["volume"]), bins=120, alpha=0.8)
    plt.title("Distribution du volume (log1p)")
    plt.xlabel("log(1+volume)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "p1_02_distributions.png"), dpi=110)
    plt.close()

    # figure 3 : matrice de correlation (1 ligne sur 10 pour aller plus vite)
    sample = prices.iloc[::10][["open", "close", "low", "high", "volume"]].copy()
    sample["log_volume"] = np.log1p(sample["volume"])
    sample = sample.drop(columns=["volume"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(sample.corr(), annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Matrice de correlation (OHLC + log volume)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "p1_03_correlation.png"), dpi=110)
    plt.close()

    # figure 4 : repartition des secteurs
    sector_by_symbol = dict(zip(securities["Ticker symbol"], securities["GICS Sector"]))
    sectors = [sector_by_symbol.get(s, "Unknown") for s in prices["symbol"].unique()]
    counts = pd.Series(sectors).value_counts().head(12)
    plt.figure(figsize=(9, 5))
    plt.barh(counts.index[::-1], counts.values[::-1])
    plt.title("Nombre d'actions par secteur GICS")
    plt.xlabel("Nombre d'actions")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "p1_04_sectors.png"), dpi=110)
    plt.close()


def prepare_data(prices):
    # pour chaque action, on prend les L derniers jours (OHLCV) pour predire le prix
    # de cloture du jour suivant
    X_list, y_list, d_list = [], [], []
    for sym, g in prices.groupby("symbol"):
        g = g.sort_values("date")
        values = g[FEATURES].to_numpy(dtype=np.float32)
        close = g["close"].to_numpy(dtype=np.float32)
        dates = g["date"].to_numpy()
        n = len(g)
        if n <= L + 1:
            continue
        # une ligne = les L jours, colonne par colonne (open des L jours, puis close, ...)
        columns = []
        for f in range(len(FEATURES)):
            for lag in range(L):
                columns.append(values[lag:n - L + lag, f])
        X_list.append(np.column_stack(columns))
        y_list.append(close[L:])
        d_list.append(dates[L:])

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    d = np.concatenate(d_list)

    # tri chronologique global
    order = np.argsort(d, kind="stable")
    X, y, d = X[order], y[order], d[order]

    # split 80% train / 20% test en respectant le temps
    cut = int(len(X) * 0.8)
    cut_date = d[cut]
    train_mask = d < cut_date
    test_mask = d >= cut_date
    X_train_raw, X_test_raw = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # standardisation (moyenne et ecart-type calcules sur le train uniquement)
    scaler_x = StandardScaler().fit(X_train_raw)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
    X_train = scaler_x.transform(X_train_raw).astype(np.float32)
    X_test = scaler_x.transform(X_test_raw).astype(np.float32)
    y_train_std = scaler_y.transform(y_train.reshape(-1, 1)).astype(np.float32).ravel()
    y_test_std = scaler_y.transform(y_test.reshape(-1, 1)).astype(np.float32).ravel()

    feature_names = ["%s_lag%d" % (f, l) for f in FEATURES for l in range(1, L + 1)]
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_train_std": y_train_std, "y_test_std": y_test_std,
        "scaler_y": scaler_y,
        "feature_names": feature_names,
        "n_train": len(X_train), "n_test": len(X_test),
        "split_date": pd.Timestamp(cut_date).date(),
    }
