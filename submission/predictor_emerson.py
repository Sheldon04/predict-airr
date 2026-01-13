## imports used by the basic code template provided.

import os
import subprocess
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import sys
import argparse
from collections import defaultdict
from typing import Iterator, Tuple, Union, List, Dict, Set

## imports that are additionally used by this notebook

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from scipy.special import betaln
from scipy.stats import betabinom as beta_binomial, fisher_exact

# ==================== Compairr helpers ====================
def _iter_repertoires(data_dir: str, metadata_filename: str = "metadata.csv"):
    """
    一个通用的 repertoire 迭代器，和你现有的 load_data_generator 逻辑相同语义：

    - 如果 metadata.csv 存在：
        yield (repertoire_id, df)
    - 否则：
        对目录下的 *.tsv 逐个读取：
        yield (basename_without_ext, df)
    """
    metadata_path = os.path.join(data_dir, metadata_filename)
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            file_path = os.path.join(data_dir, row.filename)
            rep_id = row.repertoire_id
            try:
                df = pd.read_csv(file_path, sep="\t")
            except FileNotFoundError:
                print(f"[WARN] File `{file_path}` referenced in metadata not found. Skipping.")
                continue
            yield rep_id, df
    else:
        pattern = os.path.join(data_dir, "*.tsv")
        for file_path in sorted(glob.glob(pattern)):
            rep_id = os.path.basename(file_path).replace(".tsv", "")
            try:
                df = pd.read_csv(file_path, sep="\t")
            except Exception as e:
                print(f"[WARN] Could not read `{file_path}`: {e}. Skipping.")
                continue
            yield rep_id, df


# ---------------------- 1. 构建 library.tsv（只基于训练集） ----------------------


def build_library_from_train(
    train_dir: str,
    out_path: str,
    min_repertoire_occurrence: int = 1,
) -> pd.DataFrame:
    """
    从训练集构造 CompAIRR 的 library.tsv：

    - 先在每个训练 repertoire 中按 clonotype 定义 (junction_aa, v_call, j_call) 去重；
    - 统计每个 clonotype 出现于多少个不同的 repertoires；
    - 只保留出现在 >= min_repertoire_occurrence 个 repertoires 的 clonotype；
    - 每个保留的 clonotype 在 library.tsv 中占一行。

    CompAIRR 只关心序列本身，library 的 repertoire_id 字段可以固定为 'LIB'
    （或者 1，二者对于 existence 模式都没影响）。
    """
    print(f"[build_library] Reading train repertoires from `{train_dir}`...")

    # 统计：clonotype -> set(repertoire_id)
    clone_to_reps: Dict[Tuple[str, str, str], Set[str]] = {}

    for rep_id, df in tqdm(list(_iter_repertoires(train_dir)),
                           desc="Scanning train repertoires for library"):
        if df.empty:
            continue
        if "junction_aa" not in df.columns:
            print(f"[WARN] `junction_aa` column not found in {rep_id}, skipping.")
            continue

        df = df.dropna(subset=["junction_aa"])
        if df.empty:
            continue

        # 如果没有 v_call / j_call，则用占位符（极不推荐，但防御一下）
        v_col = df["v_call"] if "v_call" in df.columns else "NA"
        j_col = df["j_call"] if "j_call" in df.columns else "NA"

        clones = set(zip(df["junction_aa"], v_col, j_col))
        for clon in clones:
            if clon not in clone_to_reps:
                clone_to_reps[clon] = set()
            clone_to_reps[clon].add(rep_id)

    print(f"[build_library] Found {len(clone_to_reps)} unique clonotypes in train set.")

    # 按 min_repertoire_occurrence 过滤
    selected_rows = []
    seq_id = 1
    for (cdr3, v, j), reps in clone_to_reps.items():
        if len(reps) < min_repertoire_occurrence:
            continue
        selected_rows.append({
            "repertoire_id": "LIB",
            "sequence_id": seq_id,
            "duplicate_count": 1,  # library 里 abundance 不重要，presence 即可
            "v_call": v,
            "j_call": j,
            "junction_aa": cdr3,
        })
        seq_id += 1

    if not selected_rows:
        print("[build_library] No clonotypes passed the min_repertoire_occurrence filter; "
              "library will be empty.")
        lib_df = pd.DataFrame(columns=[
            "repertoire_id", "sequence_id", "duplicate_count",
            "v_call", "j_call", "junction_aa"
        ])
    else:
        lib_df = pd.DataFrame(selected_rows)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lib_df.to_csv(out_path, sep="\t", index=False)
    print(f"[build_library] Wrote {len(lib_df)} clonotypes to `{out_path}`.")
    return lib_df


# ---------------------- 2. 构建 all_repertoires.tsv（train + test） ----------------------


def build_all_repertoires(
    train_dir: str,
    test_dirs: List[str],
    out_path: str,
) -> pd.DataFrame:
    """
    构造 CompAIRR B 集合输入：all_repertoires.tsv

    - 合并 train_dir + 所有 test_dirs 中的全部序列
    - 每一行对应一个 clonotype（或一条原始记录），至少包含：
        repertoire_id, sequence_id, duplicate_count, v_call, j_call, junction_aa
    """

    def _add_dir(data_dir: str, rows: List[dict], seq_counter: int) -> int:
        for rep_id, df in tqdm(list(_iter_repertoires(data_dir)),
                               desc=f"Adding repertoires from {os.path.basename(data_dir)}"):
            if df.empty:
                continue
            if "junction_aa" not in df.columns:
                print(f"[WARN] `junction_aa` column not found in {rep_id}, skipping.")
                continue

            df = df.dropna(subset=["junction_aa"])
            if df.empty:
                continue

            v_col = df["v_call"] if "v_call" in df.columns else "NA"
            j_col = df["j_call"] if "j_call" in df.columns else "NA"

            # 如果有 duplicate_count 列用之，否则认为每一行 count=1
            # if "duplicate_count" in df.columns:
            #     dup_col = df["duplicate_count"].fillna(1).astype(int)
            # else:
            #     dup_col = pd.Series([1] * len(df), index=df.index)
                # duplicate_count：优先用 duplicate_count，其次用 templates，没有就全 1
            if "duplicate_count" in df.columns:
                dup_col = df["duplicate_count"].fillna(1).astype(int)
            elif "templates" in df.columns:
                dup_col = df["templates"].fillna(1).astype(int)
            else:
                dup_col = pd.Series([1] * len(df), index=df.index, dtype=int)


            for i, row in df.iterrows():
                rows.append({
                    "repertoire_id": rep_id,
                    "sequence_id": seq_counter,
                    "duplicate_count": int(dup_col.loc[i]),
                    "v_call": row.get("v_call", "NA"),
                    "j_call": row.get("j_call", "NA"),
                    "junction_aa": row["junction_aa"],
                })
                seq_counter += 1

        return seq_counter

    print(f"[build_all_repertoires] Building all_repertoires from train + test dirs...")
    all_rows: List[dict] = []
    seq_id_counter = 1

    # 先加 train
    seq_id_counter = _add_dir(train_dir, all_rows, seq_id_counter)

    # 再加每个 test_dir
    for td in test_dirs:
        seq_id_counter = _add_dir(td, all_rows, seq_id_counter)

    if not all_rows:
        print("[build_all_repertoires] No rows found; output will be empty.")
        all_df = pd.DataFrame(columns=[
            "repertoire_id", "sequence_id", "duplicate_count",
            "v_call", "j_call", "junction_aa"
        ])
    else:
        all_df = pd.DataFrame(all_rows)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_df.to_csv(out_path, sep="\t", index=False)
    print(f"[build_all_repertoires] Wrote {len(all_df)} rows to `{out_path}`.")
    return all_df
## some utility functions such as data loaders, etc.
# ==================== Emerson / SequenceAbundance helpers ====================

def _load_metadata_df(train_dir_path: str) -> pd.DataFrame:
    metadata_path = os.path.join(train_dir_path, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise ValueError(f"Emerson baseline 需要 metadata.csv, 在 `{train_dir_path}` 未找到。")
    return pd.read_csv(metadata_path)


def _load_clonotypes_for_ids(
    data_dir: str,
    metadata_df: pd.DataFrame,
    selected_ids: List[str],
) -> Tuple[dict, dict]:
    """
    为给定的一组 repertoire_id 读取 clonotype 集合。

    返回:
        clonotypes_by_rep: {rep_id: set((junction_aa, v_call, j_call))}
        labels:            {rep_id: label_positive}
    """
    selected_ids_set = set(selected_ids)
    clonotypes_by_rep = {}
    labels = {}

    for row in metadata_df.itertuples(index=False):
        rep_id = row.repertoire_id
        if rep_id not in selected_ids_set:
            continue

        file_path = os.path.join(data_dir, row.filename)
        df = pd.read_csv(file_path, sep='\t')
        df = df.dropna(subset=['junction_aa'])

        # clonotype 定义： (junction_aa, v_call, j_call)
        clones = set(zip(df['junction_aa'], df['v_call'], df['j_call']))
        clonotypes_by_rep[rep_id] = clones
        labels[rep_id] = int(row.label_positive)

    return clonotypes_by_rep, labels


def _find_disease_associated_clonotypes(
    train_clonotypes: dict,
    train_labels: dict,
    p_value_threshold: float,
) -> Tuple[set, pd.DataFrame]:
    """
    Emerson 风格：用 one-sided Fisher 精确检验找 label-associated clonotypes。

    只用训练子集 (internal train) 的 repertoires。
    """
    counts = defaultdict(lambda: [0, 0])  # clonotype -> [pos_present, neg_present]

    N_pos = sum(1 for y in train_labels.values() if y == 1)
    N_neg = sum(1 for y in train_labels.values() if y == 0)

    # 统计 presence/absence
    for rep_id, clones in train_clonotypes.items():
        label = train_labels[rep_id]
        is_pos = (label == 1)
        for clon in clones:
            if is_pos:
                counts[clon][0] += 1
            else:
                counts[clon][1] += 1

    disease_rows = []

    for clon, (pos_present, neg_present) in counts.items():
        # 完全没出现在阳性里的 clone 不可能正关联，直接跳过
        if pos_present == 0:
            continue

        a = pos_present              # pos 中出现此 clonotype 的 repertoire 数
        c = neg_present              # neg 中出现
        b = N_pos - a
        d = N_neg - c

        # one-sided Fisher, alternative='greater'（阳性富集）
        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')

        if p <= p_value_threshold:
            disease_rows.append({
                'junction_aa': clon[0],
                'v_call': clon[1],
                'j_call': clon[2],
                'pos_count': pos_present,
                'neg_count': neg_present,
                'p_value': p,
            })

    if not disease_rows:
        disease_df = pd.DataFrame(
            columns=['junction_aa', 'v_call', 'j_call', 'pos_count', 'neg_count', 'p_value']
        )
        disease_set = set()
    else:
        disease_df = pd.DataFrame(disease_rows)
        disease_df = disease_df.sort_values('p_value', ascending=True).reset_index(drop=True)
        disease_set = set(
            zip(disease_df['junction_aa'], disease_df['v_call'], disease_df['j_call'])
        )

    return disease_set, disease_df


def _build_emerson_features(
    clonotypes_by_rep: dict,
    disease_clonotypes: set,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    把每个 repertoire 映射成 (k, n):

      - n: unique clonotype 数
      - k: 在 disease_clonotypes 里的 clonotype 数
    """
    rep_ids = sorted(clonotypes_by_rep.keys())
    ks, ns = [], []

    for rep_id in rep_ids:
        clones = clonotypes_by_rep[rep_id]
        n = len(clones)
        k = sum(1 for c in clones if c in disease_clonotypes)
        ks.append(k)
        ns.append(n)

    return np.array(ks, dtype=float), np.array(ns, dtype=float), rep_ids



def load_data_generator(data_dir: str, metadata_filename='metadata.csv') -> Iterator[
    Union[Tuple[str, pd.DataFrame, bool], Tuple[str, pd.DataFrame]]]:
    """
    A generator to load immune repertoire data.

    This function operates in two modes:
    1.  If metadata is found, it yields data based on the metadata file.
    2.  If metadata is NOT found, it uses glob to find and yield all '.tsv'
        files in the directory.

    Args:
        data_dir (str): The path to the directory containing the data.

    Yields:
        An iterator of tuples. The format depends on the mode:
        - With metadata: (repertoire_id, pd.DataFrame, label_positive)
        - Without metadata: (filename, pd.DataFrame)
    """
    metadata_path = os.path.join(data_dir, metadata_filename)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            file_path = os.path.join(data_dir, row.filename)
            try:
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield row.repertoire_id, repertoire_df, row.label_positive
            except FileNotFoundError:
                print(f"Warning: File '{row.filename}' listed in metadata not found. Skipping.")
                continue
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        for file_path in sorted(tsv_files):
            try:
                filename = os.path.basename(file_path)
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield filename, repertoire_df
            except Exception as e:
                print(f"Warning: Could not read file '{file_path}'. Error: {e}. Skipping.")
                continue


def load_full_dataset(data_dir: str) -> pd.DataFrame:
    """
    Loads all TSV files from a directory and concatenates them into a single DataFrame.

    This function handles two scenarios:
    1. If metadata.csv exists, it loads data based on the metadata and adds
       'repertoire_id' and 'label_positive' columns.
    2. If metadata.csv does not exist, it loads all .tsv files and adds
       a 'filename' column as an identifier.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        pd.DataFrame: A single, concatenated DataFrame containing all the data.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    df_list = []
    data_loader = load_data_generator(data_dir=data_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        total_files = len(metadata_df)
        for rep_id, data_df, label in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = rep_id
            data_df['label_positive'] = label
            df_list.append(data_df)
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        total_files = len(glob.glob(search_pattern))
        for filename, data_df in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = os.path.basename(filename).replace(".tsv", "")
            df_list.append(data_df)

    if not df_list:
        print("Warning: No data files were loaded.")
        return pd.DataFrame()

    full_dataset_df = pd.concat(df_list, ignore_index=True)
    return full_dataset_df



def save_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t', index=False)


def get_repertoire_ids(data_dir: str) -> list:
    """
    Retrieves repertoire IDs from the metadata file or filenames in the directory.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        list: A list of repertoire IDs.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        repertoire_ids = metadata_df['repertoire_id'].tolist()
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        repertoire_ids = [os.path.basename(f).replace('.tsv', '') for f in sorted(tsv_files)]

    return repertoire_ids



def validate_dirs_and_files(train_dir: str, test_dirs: List[str], out_dir: str) -> None:
    assert os.path.isdir(train_dir), f"Train directory `{train_dir}` does not exist."
    train_tsvs = glob.glob(os.path.join(train_dir, "*.tsv"))
    assert train_tsvs, f"No .tsv files found in train directory `{train_dir}`."
    metadata_path = os.path.join(train_dir, "metadata.csv")
    assert os.path.isfile(metadata_path), f"`metadata.csv` not found in train directory `{train_dir}`."

    for test_dir in test_dirs:
        assert os.path.isdir(test_dir), f"Test directory `{test_dir}` does not exist."
        test_tsvs = glob.glob(os.path.join(test_dir, "*.tsv"))
        assert test_tsvs, f"No .tsv files found in test directory `{test_dir}`."

    try:
        os.makedirs(out_dir, exist_ok=True)
        test_file = os.path.join(out_dir, "test_write_permission.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print(f"Failed to create or write to output directory `{out_dir}`: {e}")
        sys.exit(1)


def concatenate_output_files(out_dir: str) -> None:
    """
    Concatenates all test predictions and important sequences TSV files from the output directory.

    This function finds all files matching the patterns:
    - *_test_predictions.tsv
    - *_important_sequences.tsv

    and concatenates them to match the expected output format of submissions.csv.

    Args:
        out_dir (str): Path to the output directory containing the TSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame with predictions followed by important sequences.
                     Columns: ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    """
    predictions_pattern = os.path.join(out_dir, '*_test_predictions.tsv')
    sequences_pattern = os.path.join(out_dir, '*_important_sequences.tsv')

    predictions_files = sorted(glob.glob(predictions_pattern))
    sequences_files = sorted(glob.glob(sequences_pattern))

    df_list = []

    for pred_file in predictions_files:
        try:
            df = pd.read_csv(pred_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read predictions file '{pred_file}'. Error: {e}. Skipping.")
            continue

    for seq_file in sequences_files:
        try:
            df = pd.read_csv(seq_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read sequences file '{seq_file}'. Error: {e}. Skipping.")
            continue

    if not df_list:
        print("Warning: No output files were found to concatenate.")
        concatenated_df = pd.DataFrame(
            columns=['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call'])
    else:
        concatenated_df = pd.concat(df_list, ignore_index=True)
    submissions_file = os.path.join(out_dir, 'submissions.csv')
    concatenated_df.to_csv(submissions_file, index=False)
    print(f"Concatenated output written to `{submissions_file}`.")


def get_dataset_pairs(train_dir: str, test_dir: str) -> List[Tuple[str, List[str]]]:
    """Returns list of (train_path, [test_paths]) tuples for dataset pairs."""
    test_groups = defaultdict(list)
    for test_name in sorted(os.listdir(test_dir)):
        if test_name.startswith("test_dataset_"):
            base_id = test_name.replace("test_dataset_", "").split("_")[0]
            test_groups[base_id].append(os.path.join(test_dir, test_name))

    pairs = []
    for train_name in sorted(os.listdir(train_dir)):
        if train_name.startswith("train_dataset_"):
            train_id = train_name.replace("train_dataset_", "")
            train_path = os.path.join(train_dir, train_name)
            pairs.append((train_path, test_groups.get(train_id, [])))

    return pairs



# ==================== Emerson probabilistic classifier ====================

class EmersonProbabilisticBinaryClassifier:
    """
    简化版 ProbabilisticBinaryClassifier:

    - 输入特征为 (k, n):
        k: repertoire 中 label-associated clonotype 的个数
        n: repertoire 中总的 unique clonotype 个数
    - 每个类别各拟合一个 Beta 分布，使用 method-of-moments 估计参数
    - 通过 Beta-Binomial 模型 + 先验，计算后验概率 P(y=1 | k, n)
    """

    SMALL_POSITIVE_NUMBER = 1e-10

    def __init__(self):
        self.alpha_0 = None
        self.beta_0 = None
        self.alpha_1 = None
        self.beta_1 = None
        self.N_0 = None  # 负类样本数
        self.N_1 = None  # 正类样本数

    def _fit_beta_params(self, k_is: np.ndarray, n_is: np.ndarray) -> Tuple[float, float]:
        """method-of-moments 估计 Beta 分布参数，公式与 immuneML 中完全一致。"""
        k_is = np.asarray(k_is, dtype=float)
        n_is = np.asarray(n_is, dtype=float)

        valid = n_is > 0
        k_is = k_is[valid]
        n_is = n_is[valid]

        if k_is.size == 0:
            return 1.0, 1.0

        p = k_is / n_is
        mean = p.mean()
        var = p.var()

        if var != 0:
            alpha = np.square(mean) * (1.0 - mean) / var - mean
            beta = (mean * (1.0 - mean) / var - 1.0) * (1.0 - mean)
            alpha = float(max(alpha, self.SMALL_POSITIVE_NUMBER))
            beta = float(max(beta, self.SMALL_POSITIVE_NUMBER))
        else:
            alpha, beta = 1.0, 1.0

        return alpha, beta

    def fit(self, k: np.ndarray, n: np.ndarray, y: np.ndarray):
        """
        k, n: 1D 数组；y: 0/1 标签（0=negative, 1=positive）
        """
        k = np.asarray(k, dtype=float)
        n = np.asarray(n, dtype=float)
        y = np.asarray(y, dtype=int)

        assert k.shape == n.shape == y.shape

        self.N_0 = int(np.sum(y == 0))
        self.N_1 = int(np.sum(y == 1))

        k0, n0 = k[y == 0], n[y == 0]
        k1, n1 = k[y == 1], n[y == 1]

        self.alpha_0, self.beta_0 = self._fit_beta_params(k0, n0)
        self.alpha_1, self.beta_1 = self._fit_beta_params(k1, n1)

        return self

    def _posterior_probabilities(self, k: int, n: int) -> Tuple[float, float]:
        """返回 (P(y=0|k,n), P(y=1|k,n))。"""
        if n <= 0 or self.N_0 is None or self.N_1 is None:
            return 0.5, 0.5

        # 先验： (N_l + 1) / (N_0 + N_1 + 2)
        prior_0 = (self.N_0 + 1.0) / (self.N_0 + self.N_1 + 2.0)
        prior_1 = (self.N_1 + 1.0) / (self.N_0 + self.N_1 + 2.0)

        # Beta-Binomial 似然
        like_0 = beta_binomial.pmf(k, n, self.alpha_0, self.beta_0)
        like_1 = beta_binomial.pmf(k, n, self.alpha_1, self.beta_1)

        p0 = float(like_0 * prior_0)
        p1 = float(like_1 * prior_1)

        denom = p0 + p1
        if denom <= 0 or not np.isfinite(denom):
            return 0.5, 0.5

        return p0 / denom, p1 / denom

    def predict_proba(self, k: np.ndarray, n: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        n = np.asarray(n, dtype=float)
        probs = np.zeros_like(k, dtype=float)

        for i in range(k.shape[0]):
            _, p1 = self._posterior_probabilities(int(k[i]), int(n[i]))
            probs[i] = p1

        return probs

    def predict(self, k: np.ndarray, n: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(k, n)
        return (proba >= threshold).astype(int)


def prepare_data(X_df, labels_df, id_col='ID', label_col='label_positive'):
    """
    Merge feature matrix with labels, ensuring alignment.

    Parameters:
        X_df: DataFrame with samples as rows (index contains IDs)
        labels_df: DataFrame with ID column and label column
        id_col: Name of ID column in labels_df
        label_col: Name of label column in labels_df

    Returns:
        X: Feature matrix aligned with labels
        y: Binary labels
        common_ids: IDs that were kept
    """
    if id_col in labels_df.columns:
        labels_indexed = labels_df.set_index(id_col)[label_col]
    else:
        # Assume labels_df index is already the ID
        labels_indexed = labels_df[label_col]

    common_ids = X_df.index.intersection(labels_indexed.index)

    if len(common_ids) == 0:
        raise ValueError("No common IDs found between feature matrix and labels")

    X = X_df.loc[common_ids]
    y = labels_indexed.loc[common_ids]

    print(f"Aligned {len(common_ids)} samples with labels")

    return X, y, common_ids



## Main ImmuneStatePredictor class, where this notebook fills in a baseline predictor implementation within the placeholders 
## and replaces any example code lines with actual code that makes sense

class ImmuneStatePredictor:
    """
    Emerson 风格：SequenceAbundance + ProbabilisticBinaryClassifier 的实现，
    并且像你模板里的 k-mer+LR 一样，留出一部分训练数据作为验证集。
    """

    def __init__(self,
                 n_jobs: int = 1,
                 device: str = 'cpu',
                 log_dir: str = None,
                 cache_dir: str = None,
                 p_value_threshold: float = 0.0001,
                 val_size: float = 0.2,
                 random_state: int = 123,
                 mode: str = "dev",
                 use_compairr: bool = False,
                 compairr_dir: str = "./compairr_inputs",
                 **kwargs):
        self.train_ids_ = None
        total_cores = os.cpu_count()
        if n_jobs == -1:
            self.n_jobs = total_cores
        else:
            self.n_jobs = min(n_jobs, total_cores)

        self.device = 'cpu'

        # Emerson 模型本体
        self.model = None

        # 训练中发现的 label-associated clonotypes
        self.disease_sequence_set_ = None      # set((junction_aa, v_call, j_call))
        self.disease_sequences_df_ = None      # DataFrame，包含 p_value 等信息

        # logging / cache
        self.log_dir = log_dir
        self.cache_dir = cache_dir

        # 超参数
        if isinstance(p_value_threshold, (float, int, np.floating, np.integer)):
            self.p_value_thresholds = [float(p_value_threshold)]
        else:
            self.p_value_thresholds = [float(x) for x in p_value_threshold]

        if len(self.p_value_thresholds) == 0:
            raise ValueError("p_value_threshold cannot be empty.")

        # 记录每个阈值对应的结果（可选但你要求要“保存每个p_value对应的disease_set, disease_df”）
        self.disease_sequence_sets_by_p_ = {}   # {p: set((junction_aa, v_call, j_call))}
        self.disease_sequences_dfs_by_p_ = {}   # {p: DataFrame}
        self.val_scores_by_p_ = {}              # {p: {'roc_auc':..., 'balanced_accuracy':...}}

        # 记录最终选中的最优阈值
        self.best_p_value_threshold_ = None

        self.val_size = val_size
        self.random_state = random_state

        # 训练模式： "dev" 或 "full"
        if mode not in ("dev", "full"):
            raise ValueError(f"Unknown mode '{mode}', expected 'dev' or 'full'.")
        self.mode = mode

        # 验证集性能
        self.val_score_ = None
        self.val_ids_ = None

        # 新增：是否使用 CompAIRR，CompAIRR 文件所在目录
        self.use_compairr = use_compairr
        self.compairr_dir = compairr_dir

        # 存放从 CompAIRR pairs 文件解析出来的：
        # {repertoire_id: set((junction_aa, v_call, j_call))}
        self.compairr_clonotypes_by_rep_ = None

    def _load_compairr_clonotypes_from_pairs(self, train_dir_path: str):
        """
        从 CompAIRR 的 --no-matrix 输出 (pairs TSV) 中，构造：
            self.compairr_clonotypes_by_rep_ : {rep_id: set((junction_aa, v_call, j_call))}

        假设你之前的 CompAIRR 命令类似：
            compairr --no-matrix \
                     train_dataset_1_library.tsv \
                     train_dataset_1_all_repertoires.tsv \
                     -d 1 -t 16 \
                     -o train_dataset_1_pairs.tsv

        那么这里会在 compairr_dir 下寻找：
            <dataset_name>_pairs.tsv
        其中 dataset_name = basename(train_dir_path)，例如 train_dataset_1。
        """

        dataset_name = os.path.basename(os.path.abspath(train_dir_path.rstrip(os.sep)))
        if self.compairr_dir is None:
            raise ValueError("use_compairr=True 但 compairr_dir 未指定。")

        pairs_path = os.path.join(self.compairr_dir, f"{dataset_name}_pairs.tsv")
        if not os.path.exists(pairs_path):
            raise FileNotFoundError(f"CompAIRR pairs 文件未找到: {pairs_path}")

        print(f"[CompAIRR] 读取 pairs 文件: {pairs_path}")

        # 明确按 TSV 读
        pairs_df = pd.read_csv(pairs_path, sep="\t")

        required_cols = [
            "#repertoire_id_1", "junction_aa_1", "v_call_1", "j_call_1",
            "repertoire_id_2"
        ]
        missing = [c for c in required_cols if c not in pairs_df.columns]
        if missing:
            raise ValueError(f"[CompAIRR] pairs 文件缺少列: {missing}")

        # 只保留 library(LIB) -> repertoire 的方向
        # 如果你在 library.tsv 里把 repertoire_id 写成别的（不是 LIB），这里要相应改一下
        pairs_df["#repertoire_id_1"] = pairs_df["#repertoire_id_1"].astype(str)
        lib_mask = (pairs_df["#repertoire_id_1"] == "LIB")
        pairs_df = pairs_df[lib_mask].copy()

        # repertoire_id_2 就是 Kaggle 数据里的 repertoire_id
        pairs_df["repertoire_id_2"] = pairs_df["repertoire_id_2"].astype(str)

        from collections import defaultdict
        compairr_clonotypes_by_rep = defaultdict(set)

        for row in tqdm(pairs_df.itertuples(index=False), total=len(pairs_df)):
            rep_id = row.repertoire_id_2
            cdr3 = row.junction_aa_1
            v = row.v_call_1
            j = row.j_call_1
            clon = (cdr3, v, j)
            compairr_clonotypes_by_rep[rep_id].add(clon)

        self.compairr_clonotypes_by_rep_ = dict(compairr_clonotypes_by_rep)

        print(f"[CompAIRR] 共解析到 {len(self.compairr_clonotypes_by_rep_)} "
              f"个 repertoire 的 clonotype 集合（基于近似匹配）。")


    def fit(self, train_dir_path: str):
        """
        在给定的训练数据上训练 Emerson 模型。

        mode == "dev":
            - 显式做一次 train/val 划分（stratified）
            - 只用 train 子集来：发现 label-associated clonotypes、拟合 Beta-Binomial
            - 在 val 子集上评估 AUC 和 balanced accuracy

        mode == "full":
            - 使用全部 repertoires 作为训练集，不再划出验证集
            - disease-associated clonotypes 和 Beta-Binomial 拟合都基于全部训练数据
        """

        metadata_df = _load_metadata_df(train_dir_path)

        rep_ids = metadata_df['repertoire_id'].tolist()
        labels = metadata_df['label_positive'].values

        # --------- train/val 划分（根据 mode 决定）---------
        if self.mode == "dev":
            train_ids, val_ids = train_test_split(
                rep_ids,
                test_size=self.val_size,
                random_state=self.random_state,
                stratify=labels
            )
            train_ids = list(train_ids)
            val_ids = list(val_ids)
            print(f"[Emerson] 模式=dev, 内部划分: train={len(train_ids)}, val={len(val_ids)} repertoires.")
        else:  # mode == "full"
            train_ids = list(rep_ids)
            val_ids = []   # 不划分验证集
            print(f"[Emerson] 模式=full, 使用全部 {len(train_ids)} 个 repertoires 参与训练。")

        # --------- 只为 train/val 子集读取 clonotypes ---------
        if self.use_compairr:
            # 1) 解析 CompAIRR pairs 输出，得到 rep_id -> set((cdr3, V, J))
            self._load_compairr_clonotypes_from_pairs(train_dir_path)

            # 2) 从 metadata 构建 label 映射
            label_map = {
                row.repertoire_id: int(row.label_positive)
                for row in metadata_df.itertuples(index=False)
            }

            train_clonotypes = {}
            train_labels = {}
            for rep_id in train_ids:
                clones = self.compairr_clonotypes_by_rep_.get(rep_id, set())
                train_clonotypes[rep_id] = clones
                train_labels[rep_id] = label_map[rep_id]

            val_clonotypes = {}
            val_labels = {}
            for rep_id in val_ids:
                clones = self.compairr_clonotypes_by_rep_.get(rep_id, set())
                val_clonotypes[rep_id] = clones
                val_labels[rep_id] = label_map[rep_id]

            print(f"[Emerson+CompAIRR] 使用 CompAIRR presence 构造 clonotypes: "
                  f"train={len(train_clonotypes)}, val={len(val_clonotypes)} repertoires.")
        else:
            # 原始 Emerson：精确 clonotype = (junction_aa, v_call, j_call)
            train_clonotypes, train_labels = _load_clonotypes_for_ids(
                train_dir_path, metadata_df, train_ids
            )
            val_clonotypes, val_labels = _load_clonotypes_for_ids(
                train_dir_path, metadata_df, val_ids
            )


        # --------- 遍历多个 p-value 阈值：找 disease clonotypes -> 拟合 -> val 评估 ---------
        has_val = (len(val_ids) > 0)

        best_metric = -np.inf
        best_p = None
        best_model = None
        best_disease_set = None
        best_disease_df = None
        best_val_score = None
        best_val_ids = None
        best_order_train = None

        for p_thr in self.p_value_thresholds:
            print(f"[Emerson] finding clonotype, p_thr={p_thr}.")
            # 1) 找 disease-associated clonotypes
            disease_set, disease_df = _find_disease_associated_clonotypes(
                train_clonotypes,
                train_labels,
                p_value_threshold=p_thr
            )

            # 保存每个阈值对应的 disease_set / disease_df
            self.disease_sequence_sets_by_p_[p_thr] = disease_set
            self.disease_sequences_dfs_by_p_[p_thr] = disease_df

            print(f"[Emerson] p_thr={p_thr} -> {len(disease_set)} disease-associated clonotypes.")

            # 2) 构造 train (k,n) 并拟合 Beta-Binomial
            k_train, n_train, order_train = _build_emerson_features(train_clonotypes, disease_set)
            y_train = np.array([train_labels[rep_id] for rep_id in order_train], dtype=int)

            tmp_model = EmersonProbabilisticBinaryClassifier()
            tmp_model.fit(k_train, n_train, y_train)

            # 3) 若有 val，则评估并选最优（以 balanced_accuracy 为“验证准确率”）
            if has_val:
                k_val, n_val, order_val = _build_emerson_features(val_clonotypes, disease_set)
                y_val = np.array([val_labels[rep_id] for rep_id in order_val], dtype=int)

                proba_val = tmp_model.predict_proba(k_val, n_val)
                val_auc = roc_auc_score(y_val, proba_val)
                val_bal_acc = balanced_accuracy_score(y_val, (proba_val >= 0.5).astype(int))

                score_dict = {'roc_auc': float(val_auc), 'balanced_accuracy': float(val_bal_acc)}
                self.val_scores_by_p_[p_thr] = score_dict

                print(f"[Emerson] p_thr={p_thr} | Validation AUC={val_auc:.4f}, "
                    f"balanced_accuracy={val_bal_acc:.4f}")

                # 以 balanced_accuracy 作为选择标准（“验证准确率最高”）
                metric = val_auc
                if metric > best_metric:
                    best_metric = metric
                    best_p = p_thr
                    best_model = tmp_model
                    best_disease_set = disease_set
                    best_disease_df = disease_df
                    best_val_score = score_dict
                    best_val_ids = order_val
                    best_order_train = order_train
            else:
                # full 模式无验证集：选第一个阈值（或你也可以改成别的策略）
                if best_p is None:
                    best_p = p_thr
                    best_model = tmp_model
                    best_disease_set = disease_set
                    best_disease_df = disease_df
                    best_val_score = None
                    best_val_ids = []
                    best_order_train = order_train

        # --------- 写回最优结果到 self ---------
        self.best_p_value_threshold_ = best_p
        self.model = best_model
        self.disease_sequence_set_ = best_disease_set
        self.disease_sequences_df_ = best_disease_df
        self.val_score_ = best_val_score
        self.val_ids_ = best_val_ids
        self.train_ids_ = best_order_train

        if has_val:
            print(f"[Emerson] Best p_thr={best_p} with balanced_accuracy={best_metric:.4f}")
        else:
            print(f"[Emerson] (full) Selected p_thr={best_p} (no validation split).")

        # --------- logging：额外写入 best_p ---------
        if (self.log_dir is not None) and has_val:
            dataset_name = os.path.basename(os.path.normpath(train_dir_path))
            os.makedirs(self.log_dir, exist_ok=True)
            summary_path = os.path.join(self.log_dir, f"{dataset_name}_summary.txt")
            with open(summary_path, "w") as f:
                f.write(f"[Emerson] Best p_value_threshold = {best_p}\n")
                f.write(f"[Emerson] Validation AUC = {self.val_score_['roc_auc']:.4f}, ")
                f.write(f"balanced_accuracy = {self.val_score_['balanced_accuracy']:.4f}\n")
            print(f"[LOG] Validation summary written to `{summary_path}`")

        print("[Emerson] Training complete.")
        return self


    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        """
        对 test 目录中的 repertoires 做预测。

        先用训练阶段找到的 disease-associated clonotypes 计算 (k, n)，
        再用 EmersonProbabilisticBinaryClassifier 给出 P(label_positive=1)。
        """
        print(f"Making predictions for data in {test_dir_path}...")

        if self.model is None or self.disease_sequence_set_ is None:
            raise RuntimeError("The Emerson model has not been fitted yet. Please call `fit` first.")

        metadata_path = os.path.join(test_dir_path, 'metadata.csv')

        rep_ids = []
        ks = []
        ns = []

        if self.use_compairr:
            # 用 CompAIRR 的 presence，不再逐个读 TSV
            if os.path.exists(metadata_path):
                meta_df = pd.read_csv(metadata_path)
                rep_list = meta_df["repertoire_id"].astype(str).tolist()
            else:
                # 没 metadata 的话，用你现有的工具函数获取文件名 ID
                rep_list = get_repertoire_ids(test_dir_path)

            for rep_id in tqdm(rep_list, desc="Encoding test repertoires (Emerson+CompAIRR)"):
                clones = self.compairr_clonotypes_by_rep_.get(rep_id, set())
                n = len(clones)
                k = sum(1 for c in clones if c in self.disease_sequence_set_)

                rep_ids.append(rep_id)
                ks.append(k)
                ns.append(n)
        else:
            # 原始 Emerson：重新读建精确 clonotype
            data_loader = load_data_generator(data_dir=test_dir_path)
            search_pattern = os.path.join(test_dir_path, '*.tsv')
            total_files = len(glob.glob(search_pattern))

            for item in tqdm(data_loader, total=total_files,
                             desc="Encoding test repertoires (Emerson)"):
                if os.path.exists(metadata_path):
                    rep_id, df, label = item
                else:
                    filename, df = item
                    rep_id = os.path.basename(filename).replace(".tsv", "")

                df = df.dropna(subset=['junction_aa'])
                clones = set(zip(df['junction_aa'], df['v_call'], df['j_call']))

                n = len(clones)
                k = sum(1 for c in clones if c in self.disease_sequence_set_)

                rep_ids.append(rep_id)
                ks.append(k)
                ns.append(n)


        ks = np.array(ks, dtype=float)
        ns = np.array(ns, dtype=float)

        probabilities = self.model.predict_proba(ks, ns)

        predictions_df = pd.DataFrame({
            'ID': rep_ids,
            'dataset': [os.path.basename(test_dir_path)] * len(rep_ids),
            'label_positive_probability': probabilities
        })

        predictions_df['junction_aa'] = -999.0
        predictions_df['v_call'] = -999.0
        predictions_df['j_call'] = -999.0

        predictions_df = predictions_df[
            ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
        ]

        print(f"Prediction complete on {len(rep_ids)} examples in {test_dir_path}.")
        return predictions_df

    # def identify_associated_sequences(self, train_dir_path: str, top_k: int = 50000) -> pd.DataFrame:
    #     """
    #     使用训练阶段得到的 disease-associated clonotypes 作为“重要序列”。

    #     用 -log10(p_value) 排序，取前 top_k 条。
    #     """
    #     dataset_name = os.path.basename(train_dir_path)

    #     if self.disease_sequences_df_ is None or self.disease_sequences_df_.empty:
    #         raise RuntimeError("No disease-associated sequences available; did you call fit()?")

    #     df = self.disease_sequences_df_.copy()
    #     df['importance_score'] = -np.log10(df['p_value'] + 1e-300)

    #     df = df.sort_values(['importance_score', 'pos_count'],
    #                         ascending=[False, False]).head(top_k)

    #     df['dataset'] = dataset_name
    #     df['ID'] = [
    #         f"{dataset_name}_seq_top_{i + 1}" for i in range(len(df))
    #     ]
    #     df['label_positive_probability'] = -999.0

    #     df = df[['ID', 'dataset', 'label_positive_probability',
    #              'junction_aa', 'v_call', 'j_call']]

    #     return df

    def identify_associated_sequences(self, train_dir_path: str, top_k: int = 50000) -> pd.DataFrame:
        """
        使用训练阶段得到的 disease-associated clonotypes 作为“重要序列”。

        用 -log10(p_value) 排序，取前 top_k 条。
        如果实际可用的序列少于 top_k，则从已有序列中随机采样（可重复）补足到 top_k 行。
        """
        dataset_name = os.path.basename(train_dir_path)

        if self.disease_sequences_df_ is None or self.disease_sequences_df_.empty:
            raise RuntimeError("No disease-associated sequences available; did you call fit()?")

        df = self.disease_sequences_df_.copy()
        df['importance_score'] = -np.log10(df['p_value'] + 1e-300)

        # 先按重要性排序
        df = df.sort_values(['importance_score', 'pos_count'],
                            ascending=[False, False])

        n = len(df)
        if n >= top_k:
            # 足够多，直接取前 top_k
            df = df.head(top_k).reset_index(drop=True)
        else:
            # 不足 top_k：从已有序列中随机采样（可重复）补足
            needed = top_k - n
            # 在 [0, n-1] 中随机选 needed 个索引，允许重复
            sampled_idx = np.random.choice(n, size=needed, replace=True)
            df_extra = df.iloc[sampled_idx].copy()
            df = pd.concat([df, df_extra], ignore_index=True)
            df = df.reset_index(drop=True)

        # 下面是固定的输出格式
        df['dataset'] = dataset_name
        df['ID'] = [
            f"{dataset_name}_seq_top_{i + 1}" for i in range(len(df))
        ]
        df['label_positive_probability'] = -999.0

        df = df[['ID', 'dataset', 'label_positive_probability',
                'junction_aa', 'v_call', 'j_call']]

        return df

def build_compairr_input(train_dir: str, test_dirs: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    dataset_name = os.path.basename(os.path.abspath(train_dir.rstrip("/")))

    library_path = os.path.join(out_dir, f"{dataset_name}_library.tsv")
    all_reps_path = os.path.join(out_dir, f"{dataset_name}_all_repertoires.tsv")

    print(f"[build_compairr_input] train_dir = {train_dir}")
    print(f"[build_compairr_input] test_dirs = {test_dirs}")
    print(f"[build_compairr_input] out_dir = {out_dir}")

    # 1) 只用训练集构建 library（候选 public clones）
    if not os.path.exists(library_path):
        build_library_from_train(
            train_dir=train_dir,
            out_path=library_path,
            min_repertoire_occurrence=1
        )

    # 2) 用 train + test 构建 all_repertoires（CompAIRR B 集合）
    if not os.path.exists(all_reps_path):
        build_all_repertoires(
            train_dir=train_dir,
            test_dirs=test_dirs,
            out_path=all_reps_path,
        )

    pairs_path = os.path.join(out_dir, f"{dataset_name}_pairs.tsv")

    if not os.path.exists(pairs_path):
        cmd = [
            "./compairr-1.13.0-linux-x86_64", "-x",
            "-d", "1",
            "-t", "16",
            "--no-matrix",
            "-p", pairs_path,
            library_path,
            all_reps_path,
        ]
        print("[build_compairr_input] Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"[build_compairr_input] Done.")

