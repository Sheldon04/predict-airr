import os
import argparse
import pandas as pd
from typing import List
from submission.predictor import ImmuneStatePredictor
from submission.predictor_multikmer import ImmuneStatePredictor as MultikmerImmuneStatePredictor
from submission.predictor_kmer import ImmuneStatePredictor as KmerImmuneStatePredictor
from submission.predictor_emerson import ImmuneStatePredictor as EmersonImmuneStatePredictor, build_compairr_input
from submission.utils import save_tsv, validate_dirs_and_files


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
    predictor.important_sequences_ = predictor.identify_associated_sequences(train_dir_path=train_dir, top_k=50000)
    seqs = predictor.important_sequences_
    if seqs is None or seqs.empty:
        raise ValueError("No important sequences available to save")

    seqs_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
    save_tsv(seqs, seqs_path)
    print(f"Important sequences written to `{seqs_path}`.")


def main(train_dir: str, test_dirs: List[str], out_dir: str, n_jobs: int, device: str) -> None:
    validate_dirs_and_files(train_dir, test_dirs, out_dir)
    predictor = ImmuneStatePredictor(n_jobs=n_jobs,
                                     device=device)  # instantiate with any other parameters as defined by you in the class
    _train_predictor(predictor, train_dir)
    predictions = _generate_predictions(predictor, test_dirs)
    _save_predictions(predictions, out_dir, train_dir)
    _save_important_sequences(predictor, out_dir, train_dir)

def main_multikmer(train_dir: str, test_dirs: List[str], out_dir: str, n_jobs: int, device: str) -> None:
    validate_dirs_and_files(train_dir, test_dirs, out_dir)
    predictor = MultikmerImmuneStatePredictor(n_jobs=n_jobs,
                                              device=device,
                                             log_dir='./logs')  
    _train_predictor(predictor, train_dir)
    predictions = _generate_predictions(predictor, test_dirs)
    _save_predictions(predictions, out_dir, train_dir)
    _save_important_sequences(predictor, out_dir, train_dir)

def main_kmer(train_dir: str, test_dirs: List[str], out_dir: str, n_jobs: int, device: str) -> None:
    validate_dirs_and_files(train_dir, test_dirs, out_dir)
    predictor = KmerImmuneStatePredictor(n_jobs=n_jobs,
                                         device=device,
                                         log_dir='./logs')  
    _train_predictor(predictor, train_dir)
    predictions = _generate_predictions(predictor, test_dirs)
    _save_predictions(predictions, out_dir, train_dir)
    _save_important_sequences(predictor, out_dir, train_dir)

def main_emerson(train_dir: str, test_dirs: List[str], out_dir: str, n_jobs: int, device: str) -> None:
    validate_dirs_and_files(train_dir, test_dirs, out_dir)
    
    build_compairr_input(train_dir, test_dirs, os.path.join(out_dir, "compairr_inputs"))

    predictor = EmersonImmuneStatePredictor(
                    n_jobs=n_jobs,
                    device=device,
                    log_dir='./logs',
                    mode='dev',
                    p_value_threshold=0.0001,
                    use_compairr=True, 
                    compairr_dir=os.path.join(out_dir, "compairr_inputs")
                )
    
    _train_predictor(predictor, train_dir)
    predictions = _generate_predictions(predictor, test_dirs)
    _save_predictions(predictions, out_dir, train_dir)
    _save_important_sequences(predictor, out_dir, train_dir)


def run():
    parser = argparse.ArgumentParser(description="Immune State Predictor CLI")
    parser.add_argument("--train_dir", required=True, help="Path to training data directory")
    parser.add_argument("--test_dirs", required=True, nargs="+", help="Path(s) to test data director(ies)")
    parser.add_argument("--out_dir", type=str, default="./results", help="Path to output directory")
    parser.add_argument("--predictor", type=str, default="reproduce", help="Predictor used: emerson, kmer or multikmer (default: reproduce best predictor based on dataset)")
    parser.add_argument("--n_jobs", type=int, default=8,
                        help="Number of CPU cores to use. Use -1 for all available cores.")
    parser.add_argument("--device", type=str, default='cpu', choices=['cpu', 'cuda'],
                        help="Device to use for computation ('cpu' or 'cuda').")
    args = parser.parse_args()

    dataset_name = os.path.basename(os.path.abspath(args.train_dir.rstrip(os.sep)))
    # if args.predictor == "reproduce", this function will choose the best predictor according to the dataset_name in phase1
    if args.predictor == "reproduce":
        if dataset_name in ["train_dataset_1", "train_dataset_3", "train_dataset_6"]:
            args.predictor = "emerson"
        elif dataset_name in ["train_dataset_2", "train_dataset_4", "train_dataset_5", "train_dataset_7"]:
            args.predictor = "kmer"
        elif dataset_name in ["train_dataset_8"]:
            args.predictor = "multikmer"
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}. Cannot determine predictor type.")
    
    if args.predictor == "emerson":
        main_emerson(args.train_dir, args.test_dirs, args.out_dir, args.n_jobs, args.device)
    elif args.predictor == "kmer":
        main_kmer(args.train_dir, args.test_dirs, args.out_dir, args.n_jobs, args.device)
    elif args.predictor == "multikmer":
        main_multikmer(args.train_dir, args.test_dirs, args.out_dir, args.n_jobs, args.device)
    else:
        raise ValueError(f"Unknown predictor type: {args.predictor}. Choose from 'emerson', 'kmer', or 'multikmer'.")


if __name__ == "__main__":
    run()
    # main_emerson("/mnt/sda/Kaggle/AIRR-ML/train_datasets/train_datasets/train_dataset_1", 
    #              ["/mnt/sda/Kaggle/AIRR-ML/test_datasets/test_datasets/test_dataset_1"], 
    #              "./results", 16, 'cpu')
    
    # main_kmer("/mnt/sda/Kaggle/AIRR-ML/train_datasets/train_datasets/train_dataset_2", 
    #             ["/mnt/sda/Kaggle/AIRR-ML/test_datasets/test_datasets/test_dataset_2"], 
    #             "./results", 16, 'cpu')
    
    # main_multikmer("/mnt/sda/Kaggle/AIRR-ML/train_datasets/train_datasets/train_dataset_8", 
    #             ["/mnt/sda/Kaggle/AIRR-ML/test_datasets/test_datasets/test_dataset_8_1",
    #              "/mnt/sda/Kaggle/AIRR-ML/test_datasets/test_datasets/test_dataset_8_2", 
    #              "/mnt/sda/Kaggle/AIRR-ML/test_datasets/test_datasets/test_dataset_8_3", ], 
    #             "./results", 16, 'cpu')