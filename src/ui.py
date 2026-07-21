"""Local operator UI for the AudioShield middleware."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import settings  # noqa: E402
from middleware import process_audio, process_transcript  # noqa: E402


st.set_page_config(page_title="AudioShield", page_icon="🛡️", layout="wide")
st.title("🛡️ AudioShield")
st.caption("Context-aware security gateway for voice-AI pipelines")

with st.sidebar:
    st.header("Model gateway")
    provider_name = st.selectbox(
        "Provider", ["ollama", "openai-compatible"],
        index=0 if settings.llm_provider == "ollama" else 1,
    )
    model_name = st.text_input("Model", value=settings.llm_model)
    base_url = st.text_input("Base URL", value=settings.llm_base_url)
    api_key = st.text_input("API key", type="password", value="")
    st.caption("Ollama is local. OpenAI-compatible works with hosted APIs, vLLM, or LM Studio.")

runtime_settings = replace(
    settings,
    llm_provider=provider_name,
    llm_model=model_name,
    llm_base_url=base_url,
    llm_api_key=api_key or None,
)

analyze_tab, audit_tab = st.tabs(["Analyze", "Audit log"])

with analyze_tab:
    input_mode = st.radio("Input", ["Audio", "Transcript"], horizontal=True)
    uploaded = None
    transcript = ""
    if input_mode == "Audio":
        uploaded = st.file_uploader("Upload WAV or MP3", type=["wav", "mp3", "m4a", "flac"])
    else:
        transcript = st.text_area("Transcript", height=160)

    if st.button("Run secure pipeline", type="primary", width="stretch"):
        try:
            with st.spinner("Transcribing, generating, and checking…"):
                if input_mode == "Audio":
                    if uploaded is None:
                        st.warning("Upload an audio file first.")
                        st.stop()
                    suffix = Path(uploaded.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                        temp.write(uploaded.getbuffer())
                        temp_path = temp.name
                    try:
                        result = process_audio(temp_path, cfg=runtime_settings)
                    finally:
                        Path(temp_path).unlink(missing_ok=True)
                else:
                    result = process_transcript(transcript, cfg=runtime_settings)

            colors = {"ALLOW": "green", "BLOCK": "red", "MITIGATE": "orange"}
            st.markdown(f"### Decision: :{colors[result.decision]}[{result.decision}]")
            st.write(result.reason)
            st.subheader("Safe response")
            st.write(result.response)

            c1, c2, c3 = st.columns(3)
            c1.metric("Input risk", f"{result.input_unsafe_probability:.1%}")
            c2.metric(
                "Output risk",
                "N/A" if result.output_unsafe_probability is None
                else f"{result.output_unsafe_probability:.1%}",
            )
            c3.metric(
                "Context similarity",
                "N/A" if result.context_similarity is None
                else f"{result.context_similarity:.3f}",
            )
            with st.expander("Operator details"):
                st.json(result.to_dict())
        except Exception as exc:
            st.error(str(exc))

with audit_tab:
    log_file = Path(settings.log_path)
    if not log_file.exists():
        st.info("No security events have been logged yet.")
    else:
        records = []
        for line in log_file.read_text(encoding="utf-8").splitlines()[-100:]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if records:
            frame = pd.DataFrame(records).iloc[::-1]
            visible = [
                column for column in
                ["timestamp", "request_id", "decision", "reason", "provider", "model"]
                if column in frame.columns
            ]
            st.dataframe(frame[visible], width="stretch", hide_index=True)
            st.download_button(
                "Download JSONL audit log",
                log_file.read_bytes(),
                file_name="security_events.jsonl",
                mime="application/x-ndjson",
            )
