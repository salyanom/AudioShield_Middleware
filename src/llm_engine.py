import ollama


def generate_response(transcript):

    prompt = f"""
You are a Voice AI assistant.

Summarize the following transcript in 2-3 concise sentences.

Transcript:
{transcript}
"""

    response = ollama.chat(
        model="phi3:mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":

    sample = """
    Artificial intelligence is transforming healthcare by assisting doctors in diagnosis and treatment planning.
    """

    print(
        generate_response(sample)
    )