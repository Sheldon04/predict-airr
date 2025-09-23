import os
import argparse
from submission.predictor import ImmuneStatePredictor
from submission.utils import save_tsv


def main(train_dir: str, test_dir: str, out_dir: str, n_jobs: int, device: str) -> None:
    predictor = ImmuneStatePredictor(n_jobs=n_jobs, device=device)  # instantiate with any other parameters as defined by you in the class
    print(f"Fitting model on examples in ` {train_dir} `...")
    predictor.fit(train_dir)
    print(f"Predicting on examples in ` {test_dir} `...")
    preds = predictor.predict_proba(test_dir)
    if preds is None or preds.empty:
        print("No predictions returned; aborting save.")
    else:
        preds_path = os.path.join(out_dir, f"{os.path.basename(test_dir)}_predictions.tsv")
        save_tsv(preds, preds_path)
        print(f"Predictions written to ` {preds_path} `.")
    seqs = predictor.important_sequences_
    if seqs is not None and not seqs.empty:
        seqs_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
        save_tsv(seqs, seqs_path)
        print(f"Important sequences written to ` {seqs_path} `.")
    else:
        print("No important sequences to save.")


def run():
    parser = argparse.ArgumentParser(description="Immune State Predictor CLI")
    parser.add_argument("--train_dir", required=True, help="Path to training data directory")
    parser.add_argument("--test_dir", required=True, help="Path to test data directory")
    parser.add_argument("--out_dir", required=True, help="Path to output directory")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of CPU cores to use. Use -1 for all available cores.")
    parser.add_argument("--device", type=str, default='cpu', choices=['cpu', 'cuda'], help="Device to use for computation ('cpu' or 'cuda').")
    args = parser.parse_args()
    main(args.train_dir, args.test_dir, args.out_dir, args.n_jobs, args.device)


if __name__ == "__main__":
    run()
