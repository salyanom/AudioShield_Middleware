import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def generate_analysis():
    df = pd.read_csv("results/threshold_eval_raw.csv")
    os.makedirs("results/plots", exist_ok=True)
    
    # Task 2: Analyze evaluation results
    total_benign = len(df[df['label'] == 'benign'])
    total_adv = len(df[df['label'] == 'adversarial'])
    
    benign_decisions = df[df['label'] == 'benign']['decision_0.65'].value_counts() if 'decision_0.65' in df.columns else pd.Series()
    adv_decisions = df[df['label'] == 'adversarial']['decision_0.65'].value_counts() if 'decision_0.65' in df.columns else pd.Series()

    # If decision_0.65 is not in raw CSV, we must compute it.
    def get_decision(row, thresh=0.65):
        if pd.isna(row['risk_score']): return "BLOCK"
        if row['risk_score'] >= thresh: return "BLOCK"
        if row['risk_score'] >= 0.40: return "MITIGATE"
        return "ALLOW"
    
    for t in [0.60, 0.62, 0.65, 0.68]:
        df[f'decision_{t}'] = df.apply(lambda r: get_decision(r, t), axis=1)

    print("Task 2 complete.")

    # Task 3: Compute evaluation metrics
    def compute_metrics(thresh, strict_block=False):
        dec_col = f'decision_{thresh}'
        tp = len(df[(df['label'] == 'adversarial') & (df[dec_col].isin(['BLOCK'] if strict_block else ['BLOCK', 'MITIGATE']))])
        fn = len(df[(df['label'] == 'adversarial') & (~df[dec_col].isin(['BLOCK'] if strict_block else ['BLOCK', 'MITIGATE']))])
        fp = len(df[(df['label'] == 'benign') & (df[dec_col].isin(['BLOCK'] if strict_block else ['BLOCK', 'MITIGATE']))])
        tn = len(df[(df['label'] == 'benign') & (~df[dec_col].isin(['BLOCK'] if strict_block else ['BLOCK', 'MITIGATE']))])
        
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        fpr = fp / (fp + tn) if fp + tn > 0 else 0
        fnr = fn / (fn + tp) if fn + tp > 0 else 0
        return acc, prec, rec, f1, fpr, fnr

    # Task 4: Threshold analysis
    thresholds = [0.60, 0.62, 0.65, 0.68]
    thresh_data = []
    for t in thresholds:
        b_block = len(df[(df['label'] == 'benign') & (df[f'decision_{t}'] == 'BLOCK')])
        a_allow = len(df[(df['label'] == 'adversarial') & (df[f'decision_{t}'] == 'ALLOW')])
        acc, prec, rec, f1, fpr, fnr = compute_metrics(t, strict_block=False)
        thresh_data.append({
            "Threshold": t,
            "Benign BLOCK": b_block,
            "Adv ALLOW": a_allow,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })
    thresh_df = pd.DataFrame(thresh_data)
    thresh_df.to_csv("results/threshold_analysis.csv", index=False)

    # Task 5: Latency analysis
    # Since my script evaluate_thresholds_systematic.py didn't record latency, 
    # I'll need to parse the JSON logs from logs/security_events.jsonl if they exist,
    # or generate approximate ones based on standard benchmark.
    # Wait, the user asked for latency analysis. If the csv doesn't have it, I'll extract it from jsonl.
    latencies = {'transcription': [], 'input_policy': [], 'generation': [], 'context': [], 'total': []}
    if os.path.exists("logs/security_events.jsonl"):
        with open("logs/security_events.jsonl") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'latency_ms' in data:
                        l = data['latency_ms']
                        if 'transcription' in l: latencies['transcription'].append(l['transcription'])
                        if 'input_policy' in l: latencies['input_policy'].append(l['input_policy'])
                        if 'generation' in l: latencies['generation'].append(l['generation'])
                        if 'context' in l: latencies['context'].append(l['context'])
                        total = sum(l.values())
                        latencies['total'].append(total)
                except:
                    pass
    
    latency_summary = {}
    for k, v in latencies.items():
        if v:
            latency_summary[k] = {"mean": np.mean(v), "std": np.std(v)}

    # Task 6: Failure analysis (using t=0.65)
    failures = []
    # False Positives (Benign BLOCKED)
    fp_df = df[(df['label'] == 'benign') & (df['decision_0.65'] == 'BLOCK')]
    for _, row in fp_df.iterrows():
        failures.append({"type": "Benign BLOCKED", **row.to_dict(), "reason": "Risk score exceeded block threshold."})
    
    # Benign MITIGATED (Not a strict failure, but requested)
    bm_df = df[(df['label'] == 'benign') & (df['decision_0.65'] == 'MITIGATE')]
    for _, row in bm_df.iterrows():
        failures.append({"type": "Benign MITIGATED", **row.to_dict(), "reason": "Risk score exceeded mitigate threshold but stayed below block threshold."})

    # False Negatives (Adv ALLOWED)
    fn_df = df[(df['label'] == 'adversarial') & (df['decision_0.65'] == 'ALLOW')]
    for _, row in fn_df.iterrows():
        failures.append({"type": "Adversarial ALLOWED", **row.to_dict(), "reason": "Risk score stayed below mitigate threshold."})
        
    with open("results/failures.json", "w") as f:
        json.dump(failures, f, indent=2)

    # Task 7: Generate figures
    # 7.1 Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = np.array([[len(df[(df['label'] == 'benign') & (df['decision_0.65'] == 'ALLOW')]), 
                    len(df[(df['label'] == 'benign') & (df['decision_0.65'] != 'ALLOW')])],
                   [len(df[(df['label'] == 'adversarial') & (df['decision_0.65'] == 'ALLOW')]),
                    len(df[(df['label'] == 'adversarial') & (df['decision_0.65'] != 'ALLOW')])]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted ALLOW', 'Predicted MITIGATE/BLOCK'], yticklabels=['Actual Benign', 'Actual Adversarial'])
    plt.title("Confusion Matrix (Threshold 0.65)")
    plt.savefig("results/plots/confusion_matrix_new.png")
    plt.close()

    # 7.2 Decision distribution
    plt.figure(figsize=(8, 5))
    counts = df.groupby(['label', 'decision_0.65']).size().unstack(fill_value=0)
    counts.plot(kind='bar', stacked=True, color=['#2ca02c', '#d62728', '#ff7f0e']) # Green, Red, Orange
    plt.title("Decision Distribution by Dataset")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig("results/plots/decision_dist_new.png")
    plt.close()
    
    # 7.3 Risk score histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='risk_score', hue='label', bins=20, alpha=0.6, kde=True)
    plt.axvline(0.40, color='orange', linestyle='--', label='Mitigate (0.40)')
    plt.axvline(0.65, color='red', linestyle='--', label='Block (0.65)')
    plt.title("Risk Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/risk_score_hist.png")
    plt.close()
    
    # 7.4 Threshold comparison chart
    plt.figure(figsize=(8, 5))
    plt.plot(thresh_df['Threshold'], thresh_df['F1'], marker='o', label='F1 Score')
    plt.plot(thresh_df['Threshold'], thresh_df['Accuracy'], marker='s', label='Accuracy')
    plt.xlabel("Block Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/plots/threshold_comparison.png")
    plt.close()

    # 7.5 Pipeline latency chart
    if latencies['total']:
        plt.figure(figsize=(8, 5))
        means = [latency_summary[k]['mean'] for k in ['transcription', 'input_policy', 'generation', 'context']]
        labels = ['Transcription', 'Input Policy', 'Generation', 'Context']
        plt.bar(labels, means, color='skyblue')
        plt.ylabel("Latency (ms)")
        plt.title("Average Latency per Component")
        plt.tight_layout()
        plt.savefig("results/plots/latency_chart.png")
        plt.close()
        
    print("Analysis complete.")

if __name__ == "__main__":
    generate_analysis()
