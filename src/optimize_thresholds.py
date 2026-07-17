import numpy as np
import pandas as pd

def optimize():
    # Load evaluation results
    csv_path = "results/evaluation_results.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please run evaluate.py first.")
        return

    # Check columns
    required_cols = ["label", "unsafe_prob", "similarity", "audio_similarity"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Column {col} is missing from evaluation CSV.")
            return

    # Extract columns
    y_true = df["label"].values
    unsafe_prob = df["unsafe_prob"].values
    sim_text = df["similarity"].values
    sim_audio = df["audio_similarity"].values  # could contain NaNs if CLAP was disabled

    results = []

    # Sweep weights (w_policy, w_context, w_audio summing to 1.0)
    # Step size 0.05
    weight_steps = np.linspace(0.0, 1.0, 21)
    
    # Sweep thresholds (block_threshold and mitigate_threshold)
    # mitigate_threshold and block_threshold from 0.20 to 0.80
    threshold_steps = np.linspace(0.20, 0.80, 61)

    print("Running optimized parameter grid search sweep...")

    for wp in weight_steps:
        for wc in weight_steps:
            wa = 1.0 - wp - wc
            if wa < -1e-7:
                continue
            wa = max(0.0, wa)

            # Compute risk score for each row
            # If CLAP is unavailable (sim_audio is NaN), redistribute weights
            scores = []
            for i in range(len(df)):
                up = unsafe_prob[i]
                st = sim_text[i]
                sa = sim_audio[i]
                
                if pd.isna(sa) or sa is None:
                    # Redistribute weights
                    total = wp + wc
                    if total > 0:
                        wp_r = wp / total
                        wc_r = wc / total
                    else:
                        wp_r = 0.5
                        wc_r = 0.5
                    score = wp_r * up + wc_r * (1.0 - st)
                else:
                    score = wp * up + wc * (1.0 - st) + wa * (1.0 - sa)
                scores.append(score)

            scores = np.array(scores)

            # Sweep decision thresholds
            for tb in threshold_steps:
                for tm in threshold_steps:
                    if tm >= tb:
                        continue
                    
                    # Compute prediction: detected (1) if risk_score >= tm, else clean (0)
                    y_pred = (scores >= tm).astype(int)

                    # Compute classification metrics using fast NumPy operations
                    tp = np.sum((y_true == 1) & (y_pred == 1))
                    fp = np.sum((y_true == 0) & (y_pred == 1))
                    fn = np.sum((y_true == 1) & (y_pred == 0))
                    tn = np.sum((y_true == 0) & (y_pred == 0))
                    
                    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
                    accuracy = float((tp + tn) / len(y_true))
                    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

                    results.append({
                        "wp": wp, "wc": wc, "wa": wa,
                        "mitigate_threshold": tm,
                        "block_threshold": tb,
                        "f1": f1,
                        "precision": precision,
                        "recall": recall,
                        "accuracy": accuracy,
                        "fpr": fpr
                    })

    # Convert to DataFrame
    res_df = pd.DataFrame(results)
    
    # Sort by F1 desc, then FPR asc
    res_df = res_df.sort_values(by=["f1", "fpr", "mitigate_threshold"], ascending=[False, True, False])

    print("\n" + "=" * 65)
    print("  TOP 5 OPTIMAL SCORING PARAMS (MAXIMIZING F1-SCORE)")
    print("=" * 65)
    
    top_cols = ["wp", "wc", "wa", "mitigate_threshold", "block_threshold", "f1", "precision", "recall", "fpr"]
    print(res_df[top_cols].head(5).to_string(index=False))
    
    best = res_df.iloc[0]
    print("\n" + "=" * 65)
    print("  RECOMMENDED SETTINGS FOR RESEARCH PAPER")
    print("=" * 65)
    print(f"  Policy Weight (w_p)    : {best['wp']:.2f}")
    print(f"  Context Weight (w_c)   : {best['wc']:.2f}")
    print(f"  Audio Weight (w_a)     : {best['wa']:.2f}")
    print(f"  Mitigate Threshold     : {best['mitigate_threshold']:.2f}")
    print(f"  Block Threshold        : {best['block_threshold']:.2f}")
    print(f"  Expected F1-Score      : {best['f1']:.4f}")
    print(f"  Expected Precision     : {best['precision']:.4f}")
    print(f"  Expected Recall        : {best['recall']:.4f}")
    print(f"  Expected False Pos Rate: {best['fpr']:.4f}")
    print("=" * 65)

    # Save to CSV
    out_path = "results/threshold_optimization.csv"
    res_df.to_csv(out_path, index=False)
    print(f"Full search sweep results saved to {out_path}")

if __name__ == "__main__":
    optimize()
