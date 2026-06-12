from audio_processor import transcribe_audio
from context_verifier import verify_context
from policy_checker import check_policy
from logger import log_event
from llm_engine import generate_response


def decision_engine(similarity, policy_score):

    risk = 0

    if similarity < 0.25:
        risk += 1

    if policy_score > 0:
        risk += 1

    if risk >= 1:
        return "BLOCK"

    return "ALLOW"


def main():

    audio_path = input("Audio File Path: ")

    transcript = transcribe_audio(audio_path)

    print("\nTranscript:")
    print(transcript)

    response = generate_response(
    transcript
)

    print("\nGenerated Response:")
    print(response)

    similarity = verify_context(
        transcript,
        response
    )

    policy_score, details = check_policy(
    response
)

    decision = decision_engine(
        similarity,
        policy_score
    )

    log_event(
    audio_path,
    transcript,
    response,
    similarity,
    policy_score,
    details["unsafe_prob"],
    decision
    )
    print("\n========== REPORT ==========")

    print(
        f"Similarity Score  : "
        f"{similarity:.4f}"
    )

    print(
        f"Unsafe Probability: "
        f"{details['unsafe_prob']:.4f}"
    )

    print(
        f"Policy Score      : "
        f"{policy_score}"
    )

    print(
        f"Decision          : "
        f"{decision}"
    )

    print("============================")


if __name__ == "__main__":
    main()