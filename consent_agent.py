import os
import time
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from langchain_openai import ChatOpenAI


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "",
).strip()

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.5",
).strip()


if not OPENAI_API_KEY:

    raise ValueError(
        "OPENAI_API_KEY is missing."
    )


# ============================================================
# LLM
# ============================================================

llm_kwargs = {
    "model": OPENAI_MODEL,
    "api_key": OPENAI_API_KEY,
    "temperature": 0,
}

if OPENAI_BASE_URL:

    llm_kwargs[
        "base_url"
    ] = OPENAI_BASE_URL


llm = ChatOpenAI(
    **llm_kwargs
)


# ============================================================
# CONSENT SCHEMA
# ============================================================

class ConsentExtraction(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    permission_to_contact_hcp: Optional[
        bool
    ] = None

    permission_to_contact_reporter: Optional[
        bool
    ] = None

    permission_to_contact_patient: Optional[
        bool
    ] = None

    permission_to_contact_complaint: Optional[
        bool
    ] = None

    permission_to_contact_via_text: Optional[
        bool
    ] = None


# ============================================================
# STRUCTURED LLM
# ============================================================

consent_llm = llm.with_structured_output(
    ConsentExtraction
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

CONSENT_SYSTEM_PROMPT = """

You are the CONSENT AGENT.

Your ONLY responsibility is extracting CONSENT information
from the provided chatqna.

Do NOT extract contact information.

Do not return:

- names
- phone numbers
- emails
- addresses
- gender
- date of birth
- HCP details
- Patient details

Only determine the five permissions.

============================================================
PERMISSIONS
============================================================

1. Permission To Contact HCP

2. Permission To Contact Reporter

3. Permission To Contact Patient

4. Permission To Contact Complaint

5. Permission To Contact Via Text

============================================================
VALUES
============================================================

true
=
Explicit yes / explicit permission.

false
=
Explicit no / explicit refusal.

null
=
Not mentioned, unclear, ambiguous, or unanswered.

NEVER assume consent.

============================================================
NOISY DATA HANDLING
============================================================

The chatqna can contain:

- unwanted characters
- special symbols
- HTML
- broken text
- spelling mistakes
- ASR/transcription errors
- repeated information
- irrelevant conversation
- different languages
- mixed languages

Understand the semantic meaning of the conversation.

Ignore irrelevant noise.

Understand consent statements even if the language is different
or the text contains transcription noise.

============================================================
IMPORTANT RULES
============================================================

1. NEVER invent consent.

2. Providing a phone number does NOT automatically mean consent.

3. Providing an email does NOT automatically mean consent.

4. Explicit permission = true.

5. Explicit refusal = false.

6. Missing/unclear = null.

7. Do not use contact information to infer consent.

8. Do not extract contact information.

9. Return ONLY the five permission values.
"""


# ============================================================
# CONSENT AGENT
# ============================================================

def consent_agent(
    state: Dict[str, Any]
):

    chatqna = state.get(
        "chatqna",
        ""
    )

    if not chatqna.strip():

        return {
            "consent_result": {
                "permissions": {
                    "Permission To Contact HCP": None,
                    "Permission To Contact Reporter": None,
                    "Permission To Contact Patient": None,
                    "Permission To Contact Complaint": None,
                    "Permission To Contact Via Text": None,
                }
            },
            "consent_llm_time": 0.0,
        }

    print(
        "\n[CONSENT AGENT] Running LLM..."
    )

    start_time = time.perf_counter()

    try:

        result = consent_llm.invoke(
            [
                (
                    "system",
                    CONSENT_SYSTEM_PROMPT,
                ),
                (
                    "user",
                    f"""
Extract ONLY consent information from
the following chatqna.

CHATQNA
========

{chatqna}

========
""",
                ),
            ]
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        permissions = {

            "Permission To Contact HCP":
                result.permission_to_contact_hcp,

            "Permission To Contact Reporter":
                result.permission_to_contact_reporter,

            "Permission To Contact Patient":
                result.permission_to_contact_patient,

            "Permission To Contact Complaint":
                result.permission_to_contact_complaint,

            "Permission To Contact Via Text":
                result.permission_to_contact_via_text,
        }

        print(
            "[CONSENT AGENT] Completed."
        )

        print(
            "[CONSENT AGENT] Time:",
            round(
                elapsed,
                3
            ),
            "seconds",
        )

        return {

            "consent_result": {
                "permissions":
                    permissions
            },

            "consent_llm_time": round(
                elapsed,
                3
            ),
        }

    except Exception as exc:

        print(
            "[CONSENT AGENT] ERROR:",
            str(exc)
        )

        raise RuntimeError(
            f"Consent Agent failed: {exc}"
        ) from exc






# ============================================================
# STANDALONE TEST RUNNER
# ============================================================

if __name__ == "__main__":

    import json

    print("=" * 70)
    print("CONSENT AGENT - STANDALONE TEST")
    print("=" * 70)

    try:

        with open(
            "input.json",
            "r",
            encoding="utf-8",
        ) as file:

            input_data = json.load(file)

        chatqna = input_data.get(
            "chatqna",
            ""
        )

        result = consent_agent(
            {
                "chatqna": chatqna
            }
        )

        print("\nCONSENT AGENT OUTPUT:")
        print(
            json.dumps(
                result,
                indent=4,
                ensure_ascii=False,
            )
        )

        with open(
            "consent_output.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(
            "\nConsent agent output saved to: "
            "consent_output.json"
        )

    except Exception as exc:

        print(
            "\nConsent agent test failed:"
        )

        print(
            str(exc)
        )