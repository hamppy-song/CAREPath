# prediction/train_and_prediction.py
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, GroupShuffleSplit, train_test_split
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from carepath.utils import set_seed
from .config import parse_args
def load_dataset(embeddingf: str, pairf: str):
    with open(embeddingf, "rb") as fin:
        embedding_dict = pickle.load(fin)
    xs, ys, drugs, diseases = [], [], [], []
    with open(pairf, "r") as fin:
        lines = fin.readlines()
    for line in lines[1:]:
        drug, dis, label = line.strip().split("\t")
        drug = drug.strip()
        dis = dis.strip()
        label = int(label)
        key = f"{dis}__{drug}"
        if key not in embedding_dict:
            continue
        xs.append(embedding_dict[key])
        ys.append(label)
        drugs.append(drug)
        diseases.append(dis)
    return np.array(xs), np.array(ys), np.array(drugs), np.array(diseases)
def return_scores(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    aupr = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    return [acc, auc, aupr, f1]
def make_stacking_clf(base_models_cfg, seed: int, meta_C: float, stack_cv: int):
    base_models = []
    for m in base_models_cfg:
        name = m["name"]
        params = dict(m["params"])
        params["random_state"] = seed + int(m.get("seed_offset", 0))
        base_models.append((name, XGBClassifier(**params)))
    clf = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(C=meta_C, max_iter=1000),
        cv=stack_cv,
        n_jobs=-1,
    )
    return clf
def get_groups(split_type: str, drugs, diseases):
    """Return the grouping key for a split protocol (None for the random split)."""
    if split_type == "random":
        return None
    if split_type == "disease":
        return diseases
    if split_type == "drug":
        return drugs
    raise ValueError(f"Unknown split_type: {split_type}")
def make_811_split(xs, ys, groups, seed: int):
    """
    Partition the labeled pairs 8:1:1 into train / validation / test.
    The grouping key (disease or drug) is applied so that no entity crosses a
    partition boundary. The 10% test portion is set aside here and is used
    neither for hyperparameter selection nor for model fitting.
    """
    idx = np.arange(len(ys))
    if groups is None:
        tr_val_idx, te_idx = train_test_split(
            idx, test_size=0.10, random_state=seed, stratify=ys
        )
        tr_idx, val_idx = train_test_split(
            tr_val_idx, test_size=1 / 9, random_state=seed, stratify=ys[tr_val_idx]
        )
        return tr_idx, val_idx, te_idx
    # 90% (train+val) / 10% test, grouped
    gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    tr_val_idx, te_idx = next(gss.split(idx, ys, groups=groups))
    # within train+val: 1/9 -> validation (= 10% of the whole), grouped
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=1 / 9, random_state=seed)
    rel_tr, rel_val = next(
        gss_inner.split(tr_val_idx, ys[tr_val_idx], groups=groups[tr_val_idx])
    )
    tr_idx = tr_val_idx[rel_tr]
    val_idx = tr_val_idx[rel_val]
    return tr_idx, val_idx, te_idx
def get_cv_splitter(split_type: str, xs, ys, groups, n_splits: int, seed: int):
    """5-fold splitter over the merged train+validation set."""
    if split_type == "random":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return splitter.split(xs, ys)
    splitter = GroupKFold(n_splits=n_splits)
    return splitter.split(xs, ys, groups=groups)
def run_evaluation(args: dict, split_type: str):
    """
    Evaluation protocol (Supplementary Figure S5).
    1. Partition the labeled pairs 8:1:1 (train / validation / test) with the
       grouping key of the split protocol. The 10% test portion is set aside.
    2. Merge train + validation and fit the selected configuration with 5-fold
       cross-validation over this merged set, yielding five models.
    3. Score all five models on the held-out 10% test portion and report the
       mean +/- standard deviation of the five scores.
    Hyperparameters are selected beforehand (see hyperparameter search script)
    and passed in fixed; nothing is re-tuned here.
    """
    set_seed(args["seed"])
    xs, ys, drugs, diseases = load_dataset(args["embeddingf"], args["pairf"])
    groups = get_groups(split_type, drugs, diseases)
    # --- 1. 8:1:1 partition; the test portion is held out from here on ---
    tr_idx, val_idx, te_idx = make_811_split(xs, ys, groups, seed=args["seed"])
    trval_idx = np.concatenate([tr_idx, val_idx])
    X_trval, y_trval = xs[trval_idx], ys[trval_idx]
    groups_trval = None if groups is None else groups[trval_idx]
    X_test, y_test = xs[te_idx], ys[te_idx]
    drug_test, dis_test = drugs[te_idx], diseases[te_idx]
    print(
        f"[{split_type}] partition -> "
        f"train {len(tr_idx)} / val {len(val_idx)} / test {len(te_idx)} "
        f"(train+val merged: {len(trval_idx)})"
    )
    # --- 2. five models from 5-fold CV over train+validation ---
    cv_iter = get_cv_splitter(
        split_type=split_type,
        xs=X_trval, ys=y_trval,
        groups=groups_trval,
        n_splits=args["n_splits"],
        seed=args["seed"],
    )
    pred_rows = []
    fold_metric_rows = []
    fold_scores = []
    for fold, (fold_tr_idx, _) in enumerate(cv_iter, start=1):
        X_fit, y_fit = X_trval[fold_tr_idx], y_trval[fold_tr_idx]
        clf = make_stacking_clf(
            base_models_cfg=args["base_models"],
            seed=args["seed"],
            meta_C=args["meta_C"],
            stack_cv=args["stack_cv"],
        )
        clf.fit(X_fit, y_fit)
        # --- 3. every fold model is scored on the SAME held-out test portion ---
        y_prob = clf.predict_proba(X_test)[:, 1]
        eps = 1e-9
        y_prob_clip = np.clip(y_prob, eps, 1 - eps)
        y_logit = np.log(y_prob_clip / (1 - y_prob_clip))
        y_pred = (y_prob >= 0.5).astype(int)
        wrong = np.where(y_pred != y_test)[0]
        print(f"\n[Fold {fold}] fit on {len(fold_tr_idx)} pairs | wrong on test: {len(wrong)}")
        pred_rows.append(pd.DataFrame({
            "split_type": split_type,
            "fold_model": fold,
            "entity_key": [f"{d}__{dr}" for d, dr in zip(dis_test, drug_test)],
            "disease": dis_test,
            "drug": drug_test,
            "y_true": y_test.astype(int),
            "y_pred": y_pred.astype(int),
            "prob": y_prob.astype(float),
            "logit": y_logit.astype(float),
            "correct": (y_pred == y_test).astype(int),
        }))
        scores = return_scores(y_test, y_prob)
        print(
            f"MODEL-{fold} TEST-{split_type} "
            f"Acc: {scores[0]*100:.2f}% | AUROC: {scores[1]:.4f} | "
            f"AUPR: {scores[2]:.4f} | F1: {scores[3]:.4f}"
        )
        fold_metric_rows.append(
            {"fold_model": fold, "ACC": scores[0], "AUROC": scores[1], "AUPR": scores[2], "F1": scores[3]}
        )
        fold_scores.append(scores)
    fold_scores = np.array(fold_scores)
    fold_metric_df = pd.DataFrame(fold_metric_rows)
    print("\nPer-model metrics on the held-out test portion")
    print(fold_metric_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    pred_df = pd.concat(pred_rows, ignore_index=True)
    pred_out = args["pred_detail_file"].replace(".tsv", f"_{split_type}.tsv")
    pred_df.to_csv(pred_out, sep="\t", index=False)
    print("Saved per-pair predictions to:", pred_out)
    wrong_out = args["pred_detail_file"].replace(".tsv", f"_{split_type}_WRONG.tsv")
    pred_df[pred_df["correct"] == 0].to_csv(wrong_out, sep="\t", index=False)
    print("Saved WRONG cases to:", wrong_out)
    result = {
        "split_type": split_type,
        "AUC_mean": fold_scores[:, 1].mean(),   "AUC_std": fold_scores[:, 1].std(),
        "AUPRC_mean": fold_scores[:, 2].mean(), "AUPRC_std": fold_scores[:, 2].std(),
        "ACC_mean": fold_scores[:, 0].mean(),   "ACC_std": fold_scores[:, 0].std(),
        "F1_mean": fold_scores[:, 3].mean(),    "F1_std": fold_scores[:, 3].std(),
    }
    print(f"\n{split_type} results (mean +/- std over the five fold models):")
    for k, v in result.items():
        if k != "split_type":
            print(f"  {k}: {v:.4f}")
    return result
def main():
    args = parse_args()
    results = []
    for split in args["splits"]:
        print(f"\n===== split_type = {split} =====")
        results.append(run_evaluation(args, split))
    df_result = pd.DataFrame(results)
    cols = ["split_type", "AUC_mean", "AUC_std", "AUPRC_mean", "AUPRC_std",
            "ACC_mean", "ACC_std", "F1_mean", "F1_std"]
    df_result = df_result[cols]
    df_result.to_csv(args["output_file"], sep="\t", index=False)
    print("\nAll done. Results saved to", args["output_file"])
    for _, res in df_result.iterrows():
        print(
            f"TEST-{res.split_type} - "
            f"Acc:  {res.ACC_mean*100:.2f}% STD: {res.ACC_std*100:.2f}% | "
            f"AUROC: {res.AUC_mean:.4f} STD: {res.AUC_std:.4f} | "
            f"AUPR:  {res.AUPRC_mean:.4f} STD: {res.AUPRC_std:.4f} | "
            f"F1:    {res.F1_mean:.4f} STD: {res.F1_std:.4f}"
        )
if __name__ == "__main__":
    main()
