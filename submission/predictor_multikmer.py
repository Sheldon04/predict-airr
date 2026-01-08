import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import sys
import argparse
from collections import defaultdict, Counter
from typing import Iterator, Tuple, Union, List

## imports that are additionally used by this notebook

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline



## some utility functions such as data loaders, etc.

# ==================== helpers for caching by k-mer ====================

def _build_kmer_cache_single_k(data_dir: str, k: int,
                               cache_dir: str,
                               dataset_name: str) -> pd.DataFrame:
    """
    按 v0.5 风格为单个 k 构建缓存：
      feature_cache/{dataset}_k{k}_X.pkl

    列包含：
      - n_seq_total, n_unique_seq, mean_cdr3_len, std_cdr3_len
      - total_kmers
      - 各 k-mer 的相对频率（列名为原始 k-mer 字符串）

    不再缓存 metadata。
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    data_loader = load_data_generator(data_dir=data_dir)

    search_pattern = os.path.join(data_dir, '*.tsv')
    total_files = len(glob.glob(search_pattern))

    repertoire_features = []

    for item in tqdm(data_loader, total=total_files, desc=f"Encoding {k}-mers (build cache)"):
        if os.path.exists(metadata_path):
            rep_id, data_df, _ = item
        else:
            filename, data_df = item
            rep_id = os.path.basename(filename).replace(".tsv", "")

        seqs = list(data_df['junction_aa'].dropna()) if 'junction_aa' in data_df.columns else []
        n_seq_total = len(seqs)
        if n_seq_total > 0:
            unique_seqs = set(seqs)
            n_unique_seq = len(unique_seqs)
            lengths = [len(s) for s in seqs]
            mean_cdr3_len = float(np.mean(lengths))
            std_cdr3_len = float(np.std(lengths))
        else:
            n_unique_seq = 0
            mean_cdr3_len = 0.0
            std_cdr3_len = 0.0

        kmer_counts = Counter()
        total_kmers = 0
        for seq in seqs:
            L = len(seq)
            if L >= k:
                total_kmers += (L - k + 1)
                for i in range(L - k + 1):
                    kmer = seq[i:i + k]
                    kmer_counts[kmer] += 1

        if total_kmers > 0:
            kmer_freqs = {kmer: count / total_kmers for kmer, count in kmer_counts.items()}
        else:
            kmer_freqs = {}

        feature_row = {
            'ID': rep_id,
            'n_seq_total': n_seq_total,
            'n_unique_seq': n_unique_seq,
            'mean_cdr3_len': mean_cdr3_len,
            'std_cdr3_len': std_cdr3_len,
            'total_kmers': total_kmers,
        }
        feature_row.update(kmer_freqs)
        repertoire_features.append(feature_row)

        del data_df, kmer_counts, kmer_freqs, seqs

    features_df = pd.DataFrame(repertoire_features).set_index('ID').fillna(0)

    os.makedirs(cache_dir, exist_ok=True)
    cache_X_path = os.path.join(cache_dir, f"{dataset_name}_k{k}_X.pkl")
    features_df.to_pickle(cache_X_path)
    print(f"[CACHE] Built k={k} cache for `{dataset_name}` at {cache_dir}")

    return features_df


def _build_k4_cache_topN(data_dir: str,
                         cache_dir: str,
                         dataset_name: str,
                         k: int = 4,
                         topN: int = None) -> pd.DataFrame:
    """
    为 k=4/5 构建缓存，只保留全局 top-N k-mer。
    默认：
      - k=4: topN = 5000
      - k=5: topN = 100

    缓存格式：
      - total_kmers: repertoire 内所有 k-mer 窗口数
      - 列名为 k-mer 字符串，对应相对频率（count/total_kmers）

    不再缓存 metadata。
    """
    if k not in (4, 5):
        raise ValueError(f"_build_k4_cache_topN only supports k=4 or k=5, got k={k}")
    if topN is None:
        topN = 5000 if k == 4 else 500 # 100

    metadata_path = os.path.join(data_dir, 'metadata.csv')
    search_pattern = os.path.join(data_dir, '*.tsv')
    total_files = len(glob.glob(search_pattern))

    # ------- 第 1 轮：全局 k-mer 计数，选出 top-N -------
    global_kmer_counts = Counter()
    for item in tqdm(load_data_generator(data_dir=data_dir), total=total_files,
                     desc=f"Global scan for {k}-mers"):
        if os.path.exists(metadata_path):
            _, df, _ = item
        else:
            _, df = item
        if 'junction_aa' not in df.columns:
            continue
        for seq in df['junction_aa'].dropna().astype(str):
            L = len(seq)
            if L >= k:
                for i in range(L - k + 1):
                    global_kmer_counts[seq[i:i + k]] += 1

    top_set = {m for m, _ in global_kmer_counts.most_common(max(0, int(topN)))}
    print(f"[{k}-mer] `{dataset_name}`: selected top-{len(top_set)} {k}-mers for caching")

    # ------- 第 2 轮：逐 repertoire 计算 top-N k-mer 相对频率 -------
    repertoire_features = []

    for item in tqdm(load_data_generator(data_dir=data_dir), total=total_files,
                     desc=f"Encoding {k}-mers (build cache)"):
        if os.path.exists(metadata_path):
            rep_id, data_df, _ = item
        else:
            filename, data_df = item
            rep_id = os.path.basename(filename).replace(".tsv", "")

        seqs = data_df['junction_aa'].dropna().astype(str).tolist() if 'junction_aa' in data_df.columns else []
        total_kmers = 0
        kmer_counts = Counter()
        for seq in seqs:
            L = len(seq)
            if L >= k:
                total_kmers += (L - k + 1)
                for i in range(L - k + 1):
                    kmer = seq[i:i + k]
                    if kmer in top_set:
                        kmer_counts[kmer] += 1

        if total_kmers > 0:
            kmer_freqs = {kmer: count / total_kmers for kmer, count in kmer_counts.items()}
        else:
            kmer_freqs = {}

        feature_row = {
            'ID': rep_id,
            'total_kmers': total_kmers,
        }
        feature_row.update(kmer_freqs)
        repertoire_features.append(feature_row)

        del data_df, kmer_counts, kmer_freqs, seqs

    features_df = pd.DataFrame(repertoire_features).set_index('ID').fillna(0)

    os.makedirs(cache_dir, exist_ok=True)
    cache_X_path = os.path.join(cache_dir, f"{dataset_name}_k{k}_X.pkl")
    features_df.to_pickle(cache_X_path)
    print(f"[CACHE] Built k={k} cache (topN={topN}) for `{dataset_name}` at {cache_dir}")

    return features_df


def _build_stats_features(data_dir: str,
                          len_bins: Tuple[Tuple[int, int], ...],
                          cache_dir: str,
                          dataset_name: str,
                          top_vj_pairs_N: int = 200
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    构建“与 k 无关”的统计特征，并做缓存：
      - 长度分桶直方图（count / frac）
      - 多样性：total_reads, freq_top10/50/100, shannon_entropy, gini_simpson
      - 全 V/J/D 基因频率：v_{gene}_freq, j_{gene}_freq, d_{gene}_freq
      - Top-N VJ pair 频率：vj_{V}_{J}_freq（按序列计数，不按 templates 加权）
      - V/J family 频率：vf_{Vfamily}_freq, jf_{Jfamily}_freq
      - VJ family pair 频率：vjf_{Vfamily}_{Jfamily}_freq

    缓存文件：
      feature_cache/{dataset_name}_stats_X.pkl
      feature_cache/{dataset_name}_stats_meta.pkl
    """

    cache_stats_X_path = os.path.join(cache_dir, f"{dataset_name}_stats_X.pkl")
    cache_stats_meta_path = os.path.join(cache_dir, f"{dataset_name}_stats_meta.pkl")

    # 如需启用缓存，取消下面注释
    if os.path.exists(cache_stats_X_path) and os.path.exists(cache_stats_meta_path):
        stats_df = pd.read_pickle(cache_stats_X_path)
        meta_df = pd.read_pickle(cache_stats_meta_path)
        print(f"[CACHE] Loaded stats features for `{dataset_name}`")
        return stats_df, meta_df

    metadata_path = os.path.join(data_dir, 'metadata.csv')
    search_pattern = os.path.join(data_dir, '*.tsv')
    total_files = len(glob.glob(search_pattern))

    def which_bin(L: int) -> Union[str, None]:
        for (a, b) in len_bins:
            if a <= L <= b:
                return f"{a}_{b}"
        return None

    def pick_count_col(df: pd.DataFrame) -> Union[str, None]:
        cands = ['duplicate_count', 'templates', 'reads', 'read_count',
                 'umi_count', 'clone_count', 'count', 'frequency']
        for c in cands:
            if c in df.columns:
                return c
        return None

    def extract_gene_family(gene: str) -> Union[str, None]:
        """提取基因 family，例如 TRBV20-1*01 -> TRBV20。"""
        if not isinstance(gene, str) or gene == '':
            return None
        base = gene.split('*')[0]
        if '-' in base:
            return base.split('-')[0]
        return base

    # ---------- 第 1 轮：全局 VJ pair 计数（按序列计数），选出 top-N ----------

    global_vj_counts = Counter()

    for item in tqdm(load_data_generator(data_dir=data_dir),
                     total=total_files,
                     desc="Scanning global VJ pairs"):
        if os.path.exists(metadata_path):
            _, data_df, _ = item
        else:
            _, data_df = item

        if 'v_call' not in data_df.columns or 'j_call' not in data_df.columns:
            continue

        mask_vj = data_df['v_call'].notna() & data_df['j_call'].notna()
        if not mask_vj.any():
            continue

        sub_df = data_df.loc[mask_vj, ['v_call', 'j_call']]
        v_vals = sub_df['v_call'].astype(str).values
        j_vals = sub_df['j_call'].astype(str).values

        for v, j in zip(v_vals, j_vals):
            global_vj_counts[(v, j)] += 1

        del data_df, sub_df

    if top_vj_pairs_N > 0 and len(global_vj_counts) > 0:
        top_vj_pairs = [pair for pair, _ in global_vj_counts.most_common(top_vj_pairs_N)]
    else:
        top_vj_pairs = []
    top_vj_set = set(top_vj_pairs)
    print(f"[VJ] `{dataset_name}`: selected top-{len(top_vj_set)} VJ pairs for stats features")

    # ---------- 第 2 轮：逐 repertoire 构建特征 ----------

    rows = []
    meta_records = []

    for item in tqdm(load_data_generator(data_dir=data_dir),
                     total=total_files,
                     desc="Building stats features"):
        if os.path.exists(metadata_path):
            rep_id, data_df, label = item
        else:
            filename, data_df = item
            rep_id = os.path.basename(filename).replace(".tsv", "")
            label = None

        # ---- 序列与长度 ----
        seqs = data_df['junction_aa'].dropna().astype(str).tolist() if 'junction_aa' in data_df.columns else []
        n_seq_total = len(seqs)
        lengths = [len(s) for s in seqs]

        # ---- 长度分桶 ----
        lenbin_count = Counter()
        for L in lengths:
            tag = which_bin(L)
            if tag is not None:
                lenbin_count[tag] += 1
        lenbin_feats = {}
        for (a, b) in len_bins:
            tag = f"{a}_{b}"
            c = int(lenbin_count.get(tag, 0))
            lenbin_feats[f"lenbin_{tag}_count"] = c
            lenbin_feats[f"lenbin_{tag}_frac"] = (c / n_seq_total) if n_seq_total > 0 else 0.0

        # ---- 多样性 ----
        count_col = pick_count_col(data_df)
        freqs = None
        total_reads = 0.0
        if count_col is not None:
            counts_series = data_df[count_col].copy()
            if 'junction_aa' in data_df.columns:
                counts_series = counts_series[data_df['junction_aa'].notna()]
            counts = counts_series.fillna(0).astype(float).values
            total_reads = float(np.sum(counts))
            if total_reads > 0:
                freqs = counts / total_reads
        if freqs is None:
            if n_seq_total > 0:
                freqs = np.ones(n_seq_total, dtype=float) / n_seq_total
            else:
                freqs = np.array([], dtype=float)

        freq_sorted = np.sort(freqs)[::-1]

        def topk_sum(k):
            if freq_sorted.size == 0:
                return 0.0
            kk = min(k, freq_sorted.size)
            return float(np.sum(freq_sorted[:kk]))

        eps = 1e-12
        shannon = float(-np.sum(freqs * np.log(freqs + eps))) if freqs.size > 0 else 0.0
        gini_simpson = float(1.0 - np.sum(freqs ** 2)) if freqs.size > 0 else 0.0

        diversity_feats = {
            'total_reads': total_reads,
            'freq_top10': topk_sum(10),
            'freq_top50': topk_sum(50),
            'freq_top100': topk_sum(100),
            'shannon_entropy': shannon,
            'gini_simpson': gini_simpson,
        }

        # ---- V/J/D 频率（按序列计数）----
        v_freq_feats = {}
        j_freq_feats = {}
        d_freq_feats = {}

        if 'v_call' in data_df.columns and n_seq_total > 0:
            v_counts = data_df['v_call'].dropna().astype(str).value_counts()
            for v, c in v_counts.items():
                v_freq_feats[f"v_{v}_freq"] = float(c) / n_seq_total

        if 'j_call' in data_df.columns and n_seq_total > 0:
            j_counts = data_df['j_call'].dropna().astype(str).value_counts()
            for jg, c in j_counts.items():
                j_freq_feats[f"j_{jg}_freq"] = float(c) / n_seq_total

        if 'd_call' in data_df.columns and n_seq_total > 0:
            d_series = data_df['d_call'].dropna().astype(str)
            d_series = d_series[d_series.str.lower() != 'unknown']
            d_counts = d_series.value_counts()
            for d, c in d_counts.items():
                d_freq_feats[f"d_{d}_freq"] = float(c) / n_seq_total

        # ---- VJ pair & family 层面特征（按序列计数，不按 templates 加权）----
        vj_pair_feats = {}
        v_family_feats = {}
        j_family_feats = {}
        vj_family_pair_feats = {}

        if 'v_call' in data_df.columns and 'j_call' in data_df.columns and n_seq_total > 0:
            mask_vj = data_df['v_call'].notna() & data_df['j_call'].notna()
            if mask_vj.any():
                v_series = data_df.loc[mask_vj, 'v_call'].astype(str)
                j_series = data_df.loc[mask_vj, 'j_call'].astype(str)
                n_vj_seq = len(v_series)

                vj_counter = Counter()
                v_family_counter = Counter()
                j_family_counter = Counter()
                vj_family_counter = Counter()

                for v, j in zip(v_series.values, j_series.values):
                    vj_counter[(v, j)] += 1
                    v_fam = extract_gene_family(v)
                    j_fam = extract_gene_family(j)
                    if v_fam is not None:
                        v_family_counter[v_fam] += 1
                    if j_fam is not None:
                        j_family_counter[j_fam] += 1
                    if v_fam is not None and j_fam is not None:
                        vj_family_counter[(v_fam, j_fam)] += 1

                if n_vj_seq > 0:
                    # Top-N VJ 基因对频率
                    for (v, j) in top_vj_set:
                        c = vj_counter.get((v, j), 0)
                        if c > 0:
                            key = f"vj_{v}_{j}_freq"
                            vj_pair_feats[key] = float(c) / n_vj_seq

                    # V family 频率
                    for v_fam, c in v_family_counter.items():
                        v_family_feats[f"vf_{v_fam}_freq"] = float(c) / n_vj_seq

                    # J family 频率
                    for j_fam, c in j_family_counter.items():
                        j_family_feats[f"jf_{j_fam}_freq"] = float(c) / n_vj_seq

                    # VJ family pair 频率
                    for (v_fam, j_fam), c in vj_family_counter.items():
                        vj_family_pair_feats[f"vjf_{v_fam}_{j_fam}_freq"] = float(c) / n_vj_seq

        feature_row = {'ID': rep_id}
        feature_row.update(lenbin_feats)
        if count_col is not None:
            feature_row.update(diversity_feats)
        feature_row.update(v_freq_feats)
        feature_row.update(j_freq_feats)
        feature_row.update(d_freq_feats)
        feature_row.update(vj_pair_feats)
        feature_row.update(v_family_feats)
        feature_row.update(j_family_feats)
        feature_row.update(vj_family_pair_feats)

        rows.append(feature_row)

        md = {'ID': rep_id}
        if label is not None:
            md['label_positive'] = label
        meta_records.append(md)

        del data_df

    stats_df = pd.DataFrame(rows).set_index('ID').fillna(0)
    meta_df = pd.DataFrame(meta_records)

    os.makedirs(cache_dir, exist_ok=True)
    stats_df.to_pickle(cache_stats_X_path)
    meta_df.to_pickle(cache_stats_meta_path)
    print(f"[CACHE] Built stats features for `{dataset_name}` at {cache_dir}")

    return stats_df, meta_df




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
       'ID' and 'label_positive' columns.
    2. If metadata.csv does not exist, it loads all .tsv files and adds
       an 'ID' column as an identifier.

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


def load_and_encode_kmers(
        data_dir: str,
        k: int = 3,  # 兼容旧签名，不再使用
        kmer_k_list: Tuple[int, ...] = (3, 4),
        top4mer_N: int = 5000,
        len_bins: Tuple[Tuple[int, int], ...] = ((8, 11), (12, 14), (15, 18), (19, 30)),
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    多尺度 k-mer + 统计特征。
    - k=3：feature_cache/{dataset}_k3_X.pkl
    - k=4：feature_cache/{dataset}_k4_X.pkl（topN 使用 top4mer_N）
    - k=5：feature_cache/{dataset}_k5_X.pkl（topN 使用 _build_k4_cache_topN 默认的 100）
    """
    cache_dir = "./cache_multikmer"
    os.makedirs(cache_dir, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(data_dir))

    aa_letters = set('ACDEFGHIKLMNPQRSTVWY')

    # ----------------- k=3 缓存 -----------------
    features_k3_df = None
    if 3 in kmer_k_list:
        cache_k3_X_path = os.path.join(cache_dir, f"{dataset_name}_k3_X.pkl")
        if os.path.exists(cache_k3_X_path):
            features_k3_df = pd.read_pickle(cache_k3_X_path)
            print(f"[CACHE] Loaded k=3 features for `{dataset_name}`")
        else:
            features_k3_df = _build_kmer_cache_single_k(
                data_dir=data_dir,
                k=3,
                cache_dir=cache_dir,
                dataset_name=dataset_name
            )

    # ----------------- k=4 缓存（top-N 4-mers） -----------------
    features_k4_df = None
    if 4 in kmer_k_list:
        cache_k4_X_path = os.path.join(cache_dir, f"{dataset_name}_k4_X.pkl")
        if os.path.exists(cache_k4_X_path):
            features_k4_df = pd.read_pickle(cache_k4_X_path)
            print(f"[CACHE] Loaded k=4 features for `{dataset_name}`")
        else:
            features_k4_df = _build_k4_cache_topN(
                data_dir=data_dir,
                cache_dir=cache_dir,
                dataset_name=dataset_name,
                k=4,
                topN=top4mer_N
            )

    # ----------------- k=5 缓存（top-N 5-mers，默认 100） -----------------
    features_k5_df = None
    if 5 in kmer_k_list:
        cache_k5_X_path = os.path.join(cache_dir, f"{dataset_name}_k5_X.pkl")
        if os.path.exists(cache_k5_X_path):
            features_k5_df = pd.read_pickle(cache_k5_X_path)
            print(f"[CACHE] Loaded k=5 features for `{dataset_name}`")
        else:
            features_k5_df = _build_k4_cache_topN(
                data_dir=data_dir,
                cache_dir=cache_dir,
                dataset_name=dataset_name,
                k=5,
                topN=None,   # 使用函数内部默认：k=5 -> top100
            )

    # ----------------- 统计特征（长度、克隆多样性、V/J 频率） -----------------
    stats_df, stats_meta_df = _build_stats_features(
        data_dir,
        len_bins=len_bins,
        cache_dir=cache_dir,
        dataset_name=dataset_name,
    )
    meta_df = stats_meta_df

    # ----------------- 拼接所有特征 -----------------
    parts = []

    # k=3: 重命名 3-mer 特征为 k3_xxx，total_kmers -> total_kmers_k3
    if features_k3_df is not None:
        k3_df = features_k3_df.copy()
        rename_map = {}
        if 'total_kmers' in k3_df.columns:
            rename_map['total_kmers'] = 'total_kmers_k3'
        for col in k3_df.columns:
            if isinstance(col, str) and len(col) == 3 and set(col).issubset(aa_letters):
                rename_map[col] = f"k3_{col}"
        k3_df = k3_df.rename(columns=rename_map)
        parts.append(k3_df)

    # k=4: 重命名 4-mer 特征为 k4_xxxx，total_kmers -> total_kmers_k4
    if features_k4_df is not None:
        k4_df = features_k4_df.copy()
        rename_map = {}
        if 'total_kmers' in k4_df.columns:
            rename_map['total_kmers'] = 'total_kmers_k4'
        for col in k4_df.columns:
            if isinstance(col, str) and len(col) == 4 and set(col).issubset(aa_letters):
                rename_map[col] = f"k4_{col}"
        k4_df = k4_df.rename(columns=rename_map)
        parts.append(k4_df)

    # k=5: 重命名 5-mer 特征为 k5_xxxxx，total_kmers -> total_kmers_k5
    if features_k5_df is not None:
        k5_df = features_k5_df.copy()
        rename_map = {}
        if 'total_kmers' in k5_df.columns:
            rename_map['total_kmers'] = 'total_kmers_k5'
        for col in k5_df.columns:
            if isinstance(col, str) and len(col) == 5 and set(col).issubset(aa_letters):
                rename_map[col] = f"k5_{col}"
        k5_df = k5_df.rename(columns=rename_map)
        parts.append(k5_df)

    # 统计特征
    if stats_df is not None and not stats_df.empty:
        parts.append(stats_df)

    if not parts:
        print("Warning: no features constructed in load_and_encode_kmers; returning empty DataFrame.")
        return pd.DataFrame(), meta_df

    features_df = pd.concat(parts, axis=1)
    features_df = features_df.fillna(0)

    # 汇总 total_kmers（k3 + k4 + k5）
    if ('total_kmers_k3' in features_df.columns or
        'total_kmers_k4' in features_df.columns or
        'total_kmers_k5' in features_df.columns):

        tk3 = features_df['total_kmers_k3'].values if 'total_kmers_k3' in features_df.columns else 0
        tk4 = features_df['total_kmers_k4'].values if 'total_kmers_k4' in features_df.columns else 0
        tk5 = features_df['total_kmers_k5'].values if 'total_kmers_k5' in features_df.columns else 0
        features_df['total_kmers'] = tk3 + tk4 + tk5

    return features_df, meta_df




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


def generate_random_top_sequences_df(n_seq: int = 50000) -> pd.DataFrame:
    """
    Generates a random DataFrame simulating top important sequences.

    Args:
        n_seq (int): Number of sequences to generate.

    Returns:
        pd.DataFrame: A DataFrame with columns 'ID', 'dataset', 'junction_aa', 'v_call', 'j_call'.
    """
    seqs = set()
    while len(seqs) < n_seq:
        seq = ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), size=15))
        seqs.add(seq)
    data = {
        'junction_aa': list(seqs),
        'v_call': ['TRBV20-1'] * n_seq,
        'j_call': ['TRBJ2-7'] * n_seq,
        'importance_score': np.random.rand(n_seq)
    }
    return pd.DataFrame(data)


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
        None
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



## A Classifier class that implements functionality of a baseline prediction + identification of sequences that explain the labels
## This implementation will be used in the provided `ImmuneStatePredictor` class.

class KmerClassifier:
    """L1-regularized logistic regression for k-mer (relative freq + scalars) data with ensemble."""

    def __init__(self, c_values=None, cv_folds=5,
                 opt_metric='balanced_accuracy', random_state=123, n_jobs=1,
                 log_dir: str = None, dataset_name: str = None,
                 ensemble_size: int = 1):
        self.c_values = c_values
        self.cv_folds = cv_folds
        self.opt_metric = opt_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_C_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.model_ = None
        self.models_ = []
        self.feature_names_ = None
        self.val_score_ = None

        # logging
        self.log_dir = log_dir
        self.dataset_name = dataset_name

        # ensemble
        self.ensemble_size = max(1, int(ensemble_size))

    def _make_pipeline(self, C, random_state=None):
        """Create standardization + L1 logistic regression pipeline."""
        rs = self.random_state if random_state is None else random_state
        
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                penalty='l2', C=C, solver='liblinear',
                random_state=rs, max_iter=2000, class_weight='balanced'
            ))
        ])

    def _get_scorer(self):
        """Get scoring function for optimization."""
        if self.opt_metric == 'balanced_accuracy':
            return 'balanced_accuracy'
        elif self.opt_metric == 'roc_auc':
            return 'roc_auc'
        else:
            raise ValueError(f"Unknown metric: {self.opt_metric}")

    def tune_and_fit(self, X, y, val_size=0.2):
        """Perform CV tuning on train split and fit, with optional validation split."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=self.random_state, stratify=y)
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)
        scorer = self._get_scorer()

        results = []
        for C in self.c_values:
            pipeline = self._make_pipeline(C)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scorer,
                                    n_jobs=self.n_jobs)
            results.append({
                'C': C,
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            })

        self.cv_results_ = pd.DataFrame(results)
        best_idx = self.cv_results_['mean_score'].idxmax()
        self.best_C_ = float(self.cv_results_.loc[best_idx, 'C'])
        self.best_score_ = float(self.cv_results_.loc[best_idx, 'mean_score'])

        print(f"Best C: {self.best_C_} (CV {self.opt_metric}: {self.best_score_:.4f})")

        # ---------- 日志：每一折的 CV 结果 ----------
        if self.log_dir is not None and self.dataset_name is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            log_rows = []
            for _, row in self.cv_results_.iterrows():
                C_val = float(row['C'])
                scores = row['scores']
                for fold_idx, s in enumerate(scores):
                    log_rows.append({
                        'dataset': self.dataset_name,
                        'C': C_val,
                        'fold': int(fold_idx),
                        'score': float(s),
                        'metric': scorer
                    })
            cv_log_df = pd.DataFrame(log_rows)
            cv_log_path = os.path.join(self.log_dir, f"{self.dataset_name}_cv_folds.csv")
            cv_log_df.to_csv(cv_log_path, index=False)
            print(f"[LOG] CV fold scores written to `{cv_log_path}`")

        # Fit ensemble on training split with best hyperparameter
        self.models_ = []
        for m in range(self.ensemble_size):
            if self.random_state is None:
                rs = None
            else:
                rs = int(self.random_state) + m
            pipeline = self._make_pipeline(self.best_C_, random_state=rs)
            pipeline.fit(X_train, y_train)
            self.models_.append(pipeline)

        # 代表模型（用于解释等）
        self.model_ = self.models_[0]

        if X_val is not None:
            # 使用 ensemble 的平均概率评估 validation
            val_probs_ensemble = self.predict_proba(X_val)
            if scorer == 'balanced_accuracy':
                self.val_score_ = balanced_accuracy_score(y_val, (val_probs_ensemble >= 0.5).astype(int))
            else:  # roc_auc
                self.val_score_ = roc_auc_score(y_val, val_probs_ensemble)
            print(f"Validation {self.opt_metric}: {self.val_score_:.4f}")

            # ---------- 日志：summary ----------
            if self.log_dir is not None and self.dataset_name is not None:
                os.makedirs(self.log_dir, exist_ok=True)
                summary_path = os.path.join(self.log_dir, f"{self.dataset_name}_summary.txt")
                with open(summary_path, "w") as f:
                    f.write(f"best_C={self.best_C_}\n")
                    f.write(f"cv_{self.opt_metric}_mean={self.best_score_:.6f}\n")
                    f.write(f"val_{self.opt_metric}={self.val_score_:.6f}\n")
                print(f"[LOG] Validation summary written to `{summary_path}`")

        return self

    def predict_proba(self, X):
        """Predict class probabilities using ensemble average."""
        if not self.models_ and self.model_ is None:
            raise ValueError("Model not fitted.")
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.models_:
            probs = []
            for model in self.models_:
                probs.append(model.predict_proba(X)[:, 1])
            probs = np.stack(probs, axis=0)
            return probs.mean(axis=0)
        else:
            return self.model_.predict_proba(X)[:, 1]

    def predict(self, X):
        """Predict class labels based on ensemble average probability with threshold 0.5."""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def get_feature_importance(self):
        """
        Get feature importance from L1 coefficients of the representative model.

        Returns:
            pd.DataFrame with columns ['feature', 'coefficient', 'abs_coefficient']
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")

        coef = self.model_.named_steps['classifier'].coef_[0]

        if self.feature_names_ is not None:
            feature_names = self.feature_names_
        else:
            feature_names = [f"feature_{i}" for i in range(len(coef))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coef,
            'abs_coefficient': np.abs(coef)
        })

        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)

        return importance_df

    def score_all_sequences(self, sequences_df, sequence_col='junction_aa'):
        """
        Score all sequences using model coefficients of the representative model.
        现在支持多个 k（例如 k=3 与 k=4 同时存在），对每条序列：
        importance = Σ_k  Σ_{unique k-mer in seq} coef[k-mer]
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")

        # 取出缩放与系数，逆缩放以近似回到原始特征空间
        scaler = self.model_.named_steps['scaler']
        coef = self.model_.named_steps['classifier'].coef_[0]
        scale = scaler.scale_.copy()
        scale[scale == 0] = 1.0
        coef = coef / scale

        aa_letters = set('ACDEFGHIKLMNPQRSTVWY')

        # 按 k 分组收集 k-mer → coef，只收非零
        kmer_coef_by_k = defaultdict(dict)
        for idx, feat in enumerate(self.feature_names_):
            if isinstance(feat, str) and len(feat) > 0:
                # 支持我们在特征中采用的命名：k3_xxx / k4_xxxx
                if feat.startswith("k3_") and len(feat) == 3 + 1 + 3:
                    kmer = feat[3:]
                    if set(kmer).issubset(aa_letters) and coef[idx] != 0.0:
                        kmer_coef_by_k[3][kmer] = coef[idx]
                elif feat.startswith("k4_") and len(feat) == 3 + 1 + 4:
                    kmer = feat[3:]
                    if set(kmer).issubset(aa_letters) and coef[idx] != 0.0:
                        kmer_coef_by_k[4][kmer] = coef[idx]
                else:
                    # 兼容旧版：特征名直接就是 k-mer（如 'CAS' 或 'CASS'）
                    if set(feat).issubset(aa_letters) and coef[idx] != 0.0:
                        kmer_coef_by_k[len(feat)][feat] = coef[idx]

        seq_array = sequences_df[sequence_col].astype(str).values
        scores = []
        for seq in tqdm(seq_array, total=len(seq_array), desc="Scoring sequences"):
            L = len(seq)
            s = 0.0
            for kk, cmap in kmer_coef_by_k.items():
                if L < kk:
                    continue
                seen = set()
                for i in range(L - kk + 1):
                    kmer = seq[i:i+kk]
                    if kmer in seen:
                        continue
                    seen.add(kmer)
                    c = cmap.get(kmer)
                    if c is not None:
                        s += c
            scores.append(s)

        result_df = sequences_df.copy()
        result_df['importance_score'] = scores
        return result_df



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
        labels_indexed = labels_df[label_col]

    common_ids = X_df.index.intersection(labels_indexed.index)

    if len(common_ids) == 0:
        raise ValueError("No common IDs found between feature matrix and labels")

    X = X_df.loc[common_ids]
    y = labels_indexed.loc[common_ids]

    print(f"Aligned {len(common_ids)} samples with labels")

    return X, y, common_ids



## Main ImmuneStatePredictor class

class ImmuneStatePredictor:
    """
    A template for predicting immune states from TCR repertoire data.

    Participants should implement the logic for training, prediction, and
    sequence identification within this class.
    """

    def __init__(self, n_jobs: int = 1, device: str = 'cpu', log_dir: str = './log', seed: int = 123, **kwargs):
        """
        Initializes the predictor.

        Args:
            n_jobs (int): Number of CPU cores to use for parallel processing.
            device (str): The device to use for computation (e.g., 'cpu', 'cuda').
            **kwargs: Additional hyperparameters for the model.
        """
        self.train_ids_ = None
        total_cores = os.cpu_count()
        if n_jobs == -1:
            self.n_jobs = total_cores
        else:
            self.n_jobs = min(n_jobs, total_cores)
        self.device = device
        self.device = 'cpu'


        self.model = None
        self.important_sequences_ = None
        self.log_dir = log_dir

        self.seed = seed

    def fit(self, train_dir_path: str):
        """
        Trains the model on the provided training data.

        Args:
            train_dir_path (str): Path to the directory with training TSV files.

        Returns:
            self: The fitted predictor instance.
        """

        kmer_k_list = [3, 4, 5]

        X_train_df, y_train_df = load_and_encode_kmers(
                                    train_dir_path,
                                    kmer_k_list=kmer_k_list, # kmer_k_list=(3, 4),     # 使用 3-mer + 4-mer
                                    top4mer_N=5000,         # 4-mer 缓存内部会用到
                                    len_bins=((8, 11), (12, 14), (15, 18), (19, 30))
                                )
        

        X_train, y_train, train_ids = prepare_data(X_train_df, y_train_df,
                                                   id_col='ID', label_col='label_positive')

        dataset_name = os.path.basename(os.path.normpath(train_dir_path))

        c_values = [0.2, 0.1, 0.05, 0.03, 0.01, 0.005, 0.001]

        self.model = KmerClassifier(
            c_values=c_values,
            cv_folds=5,
            opt_metric='roc_auc',
            random_state=self.seed,
            n_jobs=self.n_jobs,
            log_dir=self.log_dir,
            dataset_name=dataset_name,
            ensemble_size=5  # 简单的 5 模型 ensemble
        )

        self.model.tune_and_fit(X_train, y_train)

        self.train_ids_ = train_ids

        # self.important_sequences_ = self.identify_associated_sequences(train_dir_path=train_dir_path, top_k=50000)

        print("Training complete.")
        return self

    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        """
        Predicts probabilities for examples in the provided path.

        Args:
            test_dir_path (str): Path to the directory with test TSV files.

        Returns:
            pd.DataFrame: A DataFrame with 'ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call' columns.
        """
        print(f"Making predictions for data in {test_dir_path}...")
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet. Please call `fit` first.")

        kmer_k_list = [3, 4, 5]

        X_test_df, _ = load_and_encode_kmers(
                            test_dir_path,
                            kmer_k_list=kmer_k_list, # kmer_k_list=(3, 4),
                            top4mer_N=5000,
                            len_bins=((8, 11), (12, 14), (15, 18), (19, 30))
                        )

        if self.model.feature_names_ is not None:
            X_test_df = X_test_df.reindex(columns=self.model.feature_names_, fill_value=0)

        repertoire_ids = X_test_df.index.tolist()

        probabilities = self.model.predict_proba(X_test_df)

        predictions_df = pd.DataFrame({
            'ID': repertoire_ids,
            'dataset': [os.path.basename(test_dir_path)] * len(repertoire_ids),
            'label_positive_probability': probabilities
        })

        predictions_df['junction_aa'] = -999.0
        predictions_df['v_call'] = -999.0
        predictions_df['j_call'] = -999.0

        predictions_df = predictions_df[['ID', 'dataset', 'label_positive_probability',
                                         'junction_aa', 'v_call', 'j_call']]

        print(f"Prediction complete on {len(repertoire_ids)} examples in {test_dir_path}.")
        return predictions_df

    def identify_associated_sequences(self, train_dir_path: str, top_k: int = 50000) -> pd.DataFrame:
        """
        Identifies the top "k" important sequences (rows) from the training data that best explain the labels.

        Args:
            top_k (int): The number of top sequences to return (based on some scoring mechanism).
            train_dir_path (str): Path to the directory with training TSV files.

        Returns:
            pd.DataFrame: A DataFrame with 'ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call' columns.
        """
        dataset_name = os.path.basename(train_dir_path)

        full_df = load_full_dataset(train_dir_path)
        unique_seqs = full_df[['junction_aa', 'v_call', 'j_call']].drop_duplicates()
        all_sequences_scored = self.model.score_all_sequences(unique_seqs, sequence_col='junction_aa')

        top_sequences_df = all_sequences_scored.nlargest(top_k, 'importance_score')
        top_sequences_df = top_sequences_df[['junction_aa', 'v_call', 'j_call']]
        top_sequences_df['dataset'] = dataset_name
        top_sequences_df['ID'] = range(1, len(top_sequences_df) + 1)
        top_sequences_df['ID'] = top_sequences_df['dataset'] + '_seq_top_' + top_sequences_df['ID'].astype(str)
        top_sequences_df['label_positive_probability'] = -999.0
        top_sequences_df = top_sequences_df[['ID', 'dataset', 'label_positive_probability',
                                             'junction_aa', 'v_call', 'j_call']]

        return top_sequences_df
    


## The `main` workflow that uses your implementation of the ImmuneStatePredictor class
## to train, identify important sequences and predict test labels

def _train_predictor(predictor: ImmuneStatePredictor, train_dir: str):
    """Trains the predictor on the training data."""
    print(f"Fitting model on examples in ` {train_dir} `...")
    predictor.fit(train_dir)


def _generate_predictions(predictor: ImmuneStatePredictor, test_dirs: List[str]) -> pd.DataFrame:
    """Generates predictions for all test directories and concatenates them."""
    all_preds = []
    for test_dir in test_dirs:
        print(f"Predicting on examples in ` {test_dir} `...")
        preds = predictor.predict_proba(test_dir)
        if preds is not None and not preds.empty:
            all_preds.append(preds)
        else:
            print(f"Warning: No predictions returned for {test_dir}")
    if all_preds:
        return pd.concat(all_preds, ignore_index=True)
    return pd.DataFrame()


def _save_predictions(predictions: pd.DataFrame, out_dir: str, train_dir: str) -> None:
    """Saves predictions to a TSV file."""
    if predictions.empty:
        raise ValueError("No predictions to save - predictions DataFrame is empty")

    preds_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_test_predictions.tsv")
    save_tsv(predictions, preds_path)
    print(f"Predictions written to `{preds_path}`.")


def _save_important_sequences(predictor: ImmuneStatePredictor, out_dir: str, train_dir: str) -> None:
    """Saves important sequences to a TSV file."""
    seqs = predictor.important_sequences_
    if seqs is None or seqs.empty:
        raise ValueError("No important sequences available to save")

    seqs_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
    save_tsv(seqs, seqs_path)
    print(f"Important sequences written to `{seqs_path}`.")


# def main(train_dir: str, test_dirs: List[str], out_dir: str, log_dir: str, n_jobs: int, device: str) -> None:
#     validate_dirs_and_files(train_dir, test_dirs, out_dir)
#     predictor = ImmuneStatePredictor(n_jobs=n_jobs, device=device, log_dir=log_dir, seed=123)
#     _train_predictor(predictor, train_dir)
#     predictor.important_sequences_ = predictor.identify_associated_sequences(train_dir_path=train_dir, top_k=50000)
#     predictions = _generate_predictions(predictor, test_dirs)
#     _save_predictions(predictions, out_dir, train_dir)
#     _save_important_sequences(predictor, out_dir, train_dir)


# def run():
#     parser = argparse.ArgumentParser(description="Immune State Predictor CLI")
#     parser.add_argument("--train_dir", required=True, help="Path to training data directory")
#     parser.add_argument("--test_dirs", required=True, nargs="+", help="Path(s) to test data director(ies)")
#     parser.add_argument("--out_dir", required=True, help="Path to output directory")
#     parser.add_argument("--n_jobs", type=int, default=1,
#                         help="Number of CPU cores to use. Use -1 for all available cores.")
#     parser.add_argument("--device", type=str, default='cpu', choices=['cpu', 'cuda'],
#                         help="Device to use for computation ('cpu' or 'cuda').")
#     args = parser.parse_args()
#     main(args.train_dir, args.test_dirs, args.out_dir, args.n_jobs, args.device)



# if __name__ == "__main__":
#     train_datasets_dir = "./train_datasets/train_datasets"
#     test_datasets_dir = "./test_datasets/test_datasets"
#     results_dir = "./results/debug"
#     log_dir = "./log/debug"

#     train_test_dataset_pairs = get_dataset_pairs(train_datasets_dir, test_datasets_dir)

#     train_test_dataset_pairs = [('./train_datasets/train_datasets/train_dataset_8', ['./test_datasets/test_datasets/test_dataset_8_1', './test_datasets/test_datasets/test_dataset_8_2', './test_datasets/test_datasets/test_dataset_8_3'])]


#     for train_dir, test_dirs in train_test_dataset_pairs:
#         main(train_dir=train_dir, test_dirs=test_dirs, out_dir=results_dir, log_dir=log_dir, n_jobs=16, device="cpu")

#     # concatenate_output_files(results_dir)
