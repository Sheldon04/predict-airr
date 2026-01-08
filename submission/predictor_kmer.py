## imports used by the basic code template provided.

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import sys
import argparse
from collections import defaultdict
from typing import Iterator, Tuple, Union, List

## imports that are additionally used by this notebook

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline



## some utility functions such as data loaders, etc.

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


def load_and_encode_kmers(data_dir: str,
                          k: int = 3,
                          cache_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loading and k-mer encoding of repertoire data, with optional caching of kmer_counts.

    Args:
        data_dir: Path to data directory
        k: K-mer length
        cache_dir: Path to cache directory; if provided, per-repertoire kmer_counts
                   will be cached and reused to avoid recomputation.

    Returns:
        Tuple of (encoded_features_df, metadata_df)
        metadata_df always contains 'ID', and 'label_positive' if available
    """
    from collections import Counter
    import pickle

    dataset_name = os.path.basename(os.path.normpath(data_dir))
    metadata_path = os.path.join(data_dir, 'metadata.csv')

    # ---------- 1. 准备 cache 路径（只存 kmer_counts） ----------
    kmer_cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        kmer_cache_path = os.path.join(cache_dir, f"{dataset_name}_k{k}_kmer_counts.pkl")

    per_rep_kmer_counts = {}

    # ---------- 2. 先尝试从 cache 里读 kmer_counts ----------
    if kmer_cache_path is not None and os.path.exists(kmer_cache_path):
        print(f"[CACHE] Loading k-mer counts from `{kmer_cache_path}`")
        with open(kmer_cache_path, "rb") as f:
            per_rep_kmer_counts = pickle.load(f)
    else:
        # ---------- 3. 没有 cache：真正去扫 TSV，算 kmer_counts ----------
        has_metadata = os.path.exists(metadata_path)
        data_loader = load_data_generator(data_dir=data_dir)

        search_pattern = os.path.join(data_dir, "*.tsv")
        total_files = len(glob.glob(search_pattern))

        for item in tqdm(data_loader, total=total_files, desc=f"Encoding {k}-mers"):
            if has_metadata:
                rep_id, data_df, label = item
            else:
                filename, data_df = item
                rep_id = os.path.basename(filename).replace(".tsv", "")

            counter = Counter()
            for seq in data_df["junction_aa"].dropna():
                L = len(seq)
                if L < k:
                    continue
                for i in range(L - k + 1):
                    counter[seq[i:i + k]] += 1

            # 只存 kmer_counts（普通 dict）
            per_rep_kmer_counts[rep_id] = dict(counter)

            del data_df, counter

        # ---------- 4. 把 kmer_counts 写入 cache（不包含 metadata） ----------
        if kmer_cache_path is not None:
            try:
                with open(kmer_cache_path, "wb") as f:
                    pickle.dump(per_rep_kmer_counts, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[CACHE] Saved k-mer counts to `{kmer_cache_path}`")
            except Exception as e:
                print(f"Warning: failed to save k-mer cache to `{kmer_cache_path}`: {e}")

    # ---------- 5. 每次现算 metadata_df（不缓存） ----------
    metadata_records = []
    if os.path.exists(metadata_path):
        # 训练集：metadata.csv 存在，列格式按照你原来的 loader 来
        meta_df = pd.read_csv(metadata_path)
        # 默认假设有这些列：repertoire_id, filename, label_positive
        for row in meta_df.itertuples(index=False):
            metadata_records.append({
                "ID": row.repertoire_id,
                "label_positive": row.label_positive,
            })
    else:
        # 测试集：没有 metadata.csv，只根据文件名给 ID
        search_pattern = os.path.join(data_dir, "*.tsv")
        for file_path in sorted(glob.glob(search_pattern)):
            rep_id = os.path.basename(file_path).replace(".tsv", "")
            metadata_records.append({"ID": rep_id})

    metadata_df = pd.DataFrame(metadata_records)

    # ---------- 6. 从 per_rep_kmer_counts 构建 features_df ----------
    repertoire_features = []
    for rep_id, kc in per_rep_kmer_counts.items():
        row = {"ID": rep_id}
        row.update(kc)
        repertoire_features.append(row)

    if repertoire_features:
        features_df = pd.DataFrame(repertoire_features).fillna(0).set_index("ID")
    else:
        print("Warning: no k-mer counts built.")
        features_df = pd.DataFrame().set_index("ID")

    return features_df, metadata_df



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





## A Classifier class that implements functionality of a baseline prediction + identification of sequences that explain the labels
## This implementation will be used in the provided `ImmuneStatePredictor` class provided through the code template, as shown in the next chunk.


class KmerClassifier:
    """L1-regularized logistic regression for k-mer count data."""

    def __init__(self, c_values=None, cv_folds=5,
                 opt_metric='balanced_accuracy', random_state=123, n_jobs=1,
                 log_dir: str = None, dataset_name: str = None):
        if c_values is None:
            c_values = [1, 0.1, 0.05, 0.03]
        self.c_values = c_values
        self.cv_folds = cv_folds
        self.opt_metric = opt_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_C_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.model_ = None
        self.feature_names_ = None
        self.val_score_ = None

        # logging (per-dataset, per-fold)
        self.log_dir = log_dir
        self.dataset_name = dataset_name

    def _make_pipeline(self, C):
        """Create standardization + L1 logistic regression pipeline."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                penalty='l1', C=C, solver='liblinear',
                random_state=self.random_state, max_iter=1000
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
        self.best_C_ = self.cv_results_.loc[best_idx, 'C']
        self.best_score_ = self.cv_results_.loc[best_idx, 'mean_score']

        print(f"Best C: {self.best_C_} (CV {self.opt_metric}: {self.best_score_:.4f})")

        # ---------- [LOG] 每个数据集每折的 CV 结果 ----------
        if self.log_dir is not None and self.dataset_name is not None:
            try:
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
            except Exception as e:
                print(f"Warning: failed to write CV log for dataset `{self.dataset_name}`: {e}")

        # Fit on training split with best hyperparameter
        self.model_ = self._make_pipeline(self.best_C_)
        self.model_.fit(X_train, y_train)

        if X_val is not None:
            if scorer == 'balanced_accuracy':
                self.val_score_ = balanced_accuracy_score(y_val, self.model_.predict(X_val))
            else:  # roc_auc
                self.val_score_ = roc_auc_score(y_val, self.model_.predict_proba(X_val)[:, 1])
            print(f"Validation {self.opt_metric}: {self.val_score_:.4f}")
        # ---------- 日志：summary ----------
        if self.log_dir is not None and self.dataset_name is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            summary_path = os.path.join(self.log_dir, f"{self.dataset_name}_summary.txt")
            with open(summary_path, "w") as f:
                f.write(f"best_C={self.best_C_}\n")
                f.write(f"cv_{self.opt_metric}_mean={self.best_score_:.6f}\n")
                if X_val is not None:
                    f.write(f"val_{self.opt_metric}={self.val_score_:.6f}\n")
            print(f"[LOG] Validation summary written to `{summary_path}`")
        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model_.predict_proba(X)[:, 1]

    def predict(self, X):
        """Predict class labels."""
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model_.predict(X)

    def get_feature_importance(self):
        """
        Get feature importance from L1 coefficients.

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

        Parameters:
            sequences_df: DataFrame with unique sequences
            sequence_col: Column name containing sequences

        Returns:
            DataFrame with added 'importance_score' column
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")

        # 取出缩放和系数，并映射回原始特征空间
        scaler = self.model_.named_steps['scaler']
        coef = self.model_.named_steps['classifier'].coef_[0]
        scale = scaler.scale_.copy()
        scale[scale == 0] = 1.0
        coef = coef / scale

        # 推断 k，并过滤出真正的 k-mer 特征
        aa_letters = set('ACDEFGHIKLMNPQRSTVWY')
        k = None
        for feat in self.feature_names_:
            if isinstance(feat, str) and len(feat) > 0 and set(feat).issubset(aa_letters):
                k = len(feat)
                break
        if k is None:
            raise ValueError("Could not infer k from feature names for sequence scoring.")

        # 建立 k-mer -> 系数 的字典，只保留非零系数的 k-mer
        kmer_coef = {}
        for idx, feat in enumerate(self.feature_names_):
            if (
                isinstance(feat, str)
                and len(feat) == k
                and set(feat).issubset(aa_letters)
                and coef[idx] != 0.0
            ):
                kmer_coef[feat] = coef[idx]

        scores = []
        seq_array = sequences_df[sequence_col].astype(str).values
        total_seqs = len(seq_array)

        for seq in tqdm(seq_array, total=total_seqs, desc="Scoring sequences"):
            L = len(seq)
            if L < k:
                scores.append(0.0)
                continue

            # 保持原来的“presence”语义：同一个 k-mer 在一条序列里只加一次
            seen = set()
            s = 0.0
            for i in range(L - k + 1):
                kmer = seq[i:i + k]
                if kmer in seen:
                    continue
                seen.add(kmer)
                c = kmer_coef.get(kmer)
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
    A template for predicting immune states from TCR repertoire data.

    Participants should implement the logic for training, prediction, and
    sequence identification within this class.
    """

    def __init__(self, n_jobs: int = 1, device: str = 'cpu', log_dir: str = None, cache_dir: str = "./cache_kmer", **kwargs):
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
        # --- your code starts here ---
        # Example: Store hyperparameters, the actual model, identified important sequences, etc.

        # NOTE: we encourage you to use self.n_jobs and self.device if appropriate in
        # your implementation instead of hardcoding these values because your code may later be run in an
        # environment with different hardware resources.

        self.model = None
        self.important_sequences_ = None

        # logging dir (for per-dataset CV fold log)
        self.log_dir = log_dir
        # k-mer cache dir
        self.cache_dir = cache_dir
        # --- your code ends here ---

    def fit(self, train_dir_path: str):
        """
        Trains the model on the provided training data.

        Args:
            train_dir_path (str): Path to the directory with training TSV files.

        Returns:
            self: The fitted predictor instance.
        """

        # --- your code starts here ---
        # Load the data, prepare suited representations as needed, train your model,
        # and find the top k important sequences that best explain the labels.
        # Example: Load the data. One possibility could be to use the provided utility function as shown below.

        # full_train_dataset_df = load_full_dataset(train_dir_path)

        X_train_df, y_train_df = load_and_encode_kmers(
            train_dir_path,
            k=4,
            cache_dir=self.cache_dir
        )  # Example of loading and encoding kmers


        #   Model Training
        #    Example: self.model = SomeClassifier().fit(X_train, y_train)

        X_train, y_train, train_ids = prepare_data(X_train_df, y_train_df,
                                                   id_col='ID', label_col='label_positive')

        # 数据集名，用于 log
        dataset_name = os.path.basename(os.path.normpath(train_dir_path))

        self.model = KmerClassifier(
            c_values=[1, 0.2, 0.1, 0.05, 0.03],
            cv_folds=5,
            opt_metric='roc_auc',
            random_state=123,
            n_jobs=self.n_jobs,
            log_dir=self.log_dir,
            dataset_name=dataset_name
        )

        self.model.tune_and_fit(X_train, y_train)

        self.train_ids_ = train_ids

        #   Identify important sequences (can be done here or in the dedicated method)
        #    Example:
        # self.important_sequences_ = self.identify_associated_sequences(train_dir_path=train_dir_path, top_k=50000)

        # --- your code ends here ---
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

        # --- your code starts here ---

        # Example: Load the data. One possibility could be to use the provided utility function as shown below.

        # full_test_dataset_df = load_full_dataset(test_dir_path)

        X_test_df, _ = load_and_encode_kmers(
            test_dir_path,
            k=4,
            cache_dir=self.cache_dir
        )

        if self.model.feature_names_ is not None:
            X_test_df = X_test_df.reindex(columns=self.model.feature_names_, fill_value=0)

        repertoire_ids = X_test_df.index.tolist()

        # Prediction
        #    Example:
        # draw random probabilities for demonstration purposes

        probabilities = self.model.predict_proba(X_test_df)

        # --- your code ends here ---

        predictions_df = pd.DataFrame({
            'ID': repertoire_ids,
            'dataset': [os.path.basename(test_dir_path)] * len(repertoire_ids),
            'label_positive_probability': probabilities
        })

        # to enable compatibility with the expected output format that includes junction_aa, v_call, j_call columns
        predictions_df['junction_aa'] = -999.0
        predictions_df['v_call'] = -999.0
        predictions_df['j_call'] = -999.0

        predictions_df = predictions_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

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

        # --- your code starts here ---
        # Return the top k sequences, sorted based on some form of importance score.
        # Example:
        # all_sequences_scored = self._score_all_sequences()

        full_df = load_full_dataset(train_dir_path)
        unique_seqs = full_df[['junction_aa', 'v_call', 'j_call']].drop_duplicates()
        all_sequences_scored = self.model.score_all_sequences(unique_seqs, sequence_col='junction_aa')

        # --- your code ends here ---

        top_sequences_df = all_sequences_scored.nlargest(top_k, 'importance_score')
        top_sequences_df = top_sequences_df[['junction_aa', 'v_call', 'j_call']]
        top_sequences_df['dataset'] = dataset_name
        top_sequences_df['ID'] = range(1, len(top_sequences_df)+1)
        top_sequences_df['ID'] = top_sequences_df['dataset'] + '_seq_top_' + top_sequences_df['ID'].astype(str)
        top_sequences_df['label_positive_probability'] = -999.0 # to enable compatibility with the expected output format
        top_sequences_df = top_sequences_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

        return top_sequences_df
    
