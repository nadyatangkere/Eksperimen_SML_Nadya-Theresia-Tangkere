import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suf in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def detect_target_column(df: pd.DataFrame) -> str | None:
    # Jangan pakai "y" sebagai substring (rawan salah deteksi: country/city/years)
    keys = ["churn", "churned", "target", "label", "exited", "attrition"]

    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in keys):
            return c

    # Jika memang ada kolom bernama tepat "y"
    for c in df.columns:
        if c.strip().lower() == "y":
            return c

    # fallback biner
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            continue
        uniq = pd.unique(s)
        if len(uniq) == 2:
            norm = set(str(u).strip().lower() for u in uniq)
            if norm.issubset({"0", "1", "yes", "no", "true", "false", "y", "n"}):
                return c

    return None


def coerce_binary_target(y: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(y):
        return y.to_numpy()

    s = y.astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}
    uniq = set(s.dropna().unique())

    if uniq.issubset(set(mapping.keys())):
        return s.map(mapping).to_numpy()

    return pd.factorize(s)[0]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # pakai sparse_output=True agar aman memori
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Path ke file raw")
    ap.add_argument("--out_preproc_dir", required=True, help="Folder preprocessing/ecommerce_customer_churn_preprocessing")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_path = Path(args.raw)
    out_preproc_dir = ensure_dir(Path(args.out_preproc_dir))

    df = read_table(raw_path)

    # target detection
    target_col = detect_target_column(df)
    if target_col is None:
        raise ValueError(
            "Target/label tidak terdeteksi otomatis.\n"
            "Solusi: rename kolom label agar mengandung 'churn'/'target' atau set manual di detect_target_column()."
        )

    (out_preproc_dir / "target_name.txt").write_text(target_col, encoding="utf-8")

    X = df.drop(columns=[target_col]).copy()
    y = coerce_binary_target(df[target_col])

    idx = np.arange(len(df))
    stratify = y if len(np.unique(y)) <= 20 else None

    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=stratify
    )

    # simpan split idx biar reproducible
    np.save(out_preproc_dir / "train_idx.npy", train_idx)
    np.save(out_preproc_dir / "test_idx.npy", test_idx)

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]

    preprocessor = build_preprocessor(X_train)
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # feature names
    try:
        feature_names = [str(n) for n in preprocessor.get_feature_names_out()]
    except Exception:
        feature_names = [f"f{i}" for i in range(X_train_p.shape[1])]

    (out_preproc_dir / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8"
    )

    dump(preprocessor, out_preproc_dir / "preprocessor.joblib")

    n_features = X_train_p.shape[1]
    if n_features > 5000:
        raise MemoryError(
            f"Jumlah fitur setelah OHE terlalu besar ({n_features}). "
            "CSV dense berisiko crash. Kurangi fitur kategori (mis. drop City) atau simpan sebagai parquet/npz."
        )

    X_train_dense = X_train_p.toarray() if hasattr(X_train_p, "toarray") else X_train_p
    X_test_dense = X_test_p.toarray() if hasattr(X_test_p, "toarray") else X_test_p

    train_df = pd.DataFrame(X_train_dense, columns=feature_names)
    test_df = pd.DataFrame(X_test_dense, columns=feature_names)

    train_df[target_col] = y_train
    test_df[target_col] = y_test

    train_df.to_csv(out_preproc_dir / "customer_churn_test.csv", index=False)
    test_df.to_csv(out_preproc_dir / "customer_churn_train.csv", index=False)

    full_df = pd.concat(
        [train_df.assign(split="train"), test_df.assign(split="test")],
        ignore_index=True
    )
    full_df.to_csv(out_preproc_dir / "processed_full.csv", index=False)

    print("DONE")
    print("Target:", target_col)
    print("Saved:", (out_preproc_dir / "processed_train.csv").resolve())
    print("Saved:", (out_preproc_dir / "processed_test.csv").resolve())


if __name__ == "__main__":
    main()