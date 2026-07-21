import pandas as pd
import numpy as np

def _compute_risk_score(unsafe_prob, transcript_sim, audio_sim):
    w_policy = 0.40
    w_context = 0.35
    w_audio = 0.25
    if pd.isna(audio_sim):
        total = w_policy + w_context
        w_policy = w_policy / total
        w_context = w_context / total
        w_audio = 0.0
    
    score = (
        w_policy * unsafe_prob +
        w_context * (1.0 - transcript_sim) +
        (w_audio * (1.0 - audio_sim) if not pd.isna(audio_sim) else 0.0)
    )
    return round(float(score), 4)

def main():
    df = pd.read_csv("results/threshold_eval_raw.csv")
    
    # Recompute risk_score for all rows using max(input, output)
    for idx, row in df.iterrows():
        input_prob = row['input_unsafe_probability']
        output_prob = row['output_unsafe_probability']
        if input_prob >= 0.60:
            unsafe = input_prob
        else:
            unsafe = output_prob if not pd.isna(output_prob) else input_prob
            
        df.at[idx, 'risk_score'] = _compute_risk_score(unsafe, row['context_similarity'], row['audio_similarity'])
    
    df.to_csv("results/threshold_eval_raw.csv", index=False)
    print("Recomputed risk scores using max(input_prob, output_prob).")

if __name__ == "__main__":
    main()
