import json
import os
import re
import time
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict


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
    "",
).strip()

OPENAI_TEMPERATURE = float(
    os.getenv(
        "OPENAI_TEMPERATURE",
        "0",
    )
)


if not OPENAI_API_KEY:

    raise ValueError(
        "OPENAI_API_KEY is missing."
    )


if not OPENAI_BASE_URL:

    raise ValueError(
        "OPENAI_BASE_URL is missing."
    )


if not OPENAI_MODEL:

    raise ValueError(
        "OPENAI_MODEL is missing."
    )


# ============================================================
# OPENAI-COMPATIBLE CLIENT
#
# Databricks Model Serving endpoint.
# ============================================================

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


# ============================================================
# CONSENT SCHEMA
# ============================================================

class ConsentExtraction(BaseModel):

    model_config = ConfigDict(
        extra="ignore"
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

9. Return ONLY valid JSON.

10. Do not return Markdown.

11. Do not return ```json.

12. Do not return explanations.

============================================================
OUTPUT FORMAT
============================================================

Return exactly this structure:

{
    "permission_to_contact_hcp": null,
    "permission_to_contact_reporter": null,
    "permission_to_contact_patient": null,
    "permission_to_contact_complaint": null,
    "permission_to_contact_via_text": null
}

"""


# ============================================================
# CLEAN MODEL JSON
# ============================================================

def extract_json_from_response(
    content: Any
) -> Dict[str, Any]:
    """
    Converts different possible model response formats
    into a Python dictionary.

    Supports:
        string
        list of content blocks
        dictionary
        markdown fenced JSON
    """

    # --------------------------------------------------------
    # STRING
    # --------------------------------------------------------

    if isinstance(
        content,
        str
    ):

        text = content.strip()

    # --------------------------------------------------------
    # LIST
    # --------------------------------------------------------

    elif isinstance(
        content,
        list
    ):

        parts = []

        for item in content:

            if isinstance(
                item,
                str
            ):

                parts.append(
                    item
                )

            elif isinstance(
                item,
                dict
            ):

                if item.get(
                    "type"
                ) == "text":

                    parts.append(
                        str(
                            item.get(
                                "text",
                                ""
                            )
                        )
                    )

                elif "text" in item:

                    parts.append(
                        str(
                            item[
                                "text"
                            ]
                        )
                    )

        text = "".join(
            parts
        ).strip()

    # --------------------------------------------------------
    # DICTIONARY
    # --------------------------------------------------------

    elif isinstance(
        content,
        dict
    ):

        return content

    else:

        raise ValueError(
            "Unsupported LLM response format."
        )

    # --------------------------------------------------------
    # REMOVE MARKDOWN FENCE
    # --------------------------------------------------------

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^```\s*",
        "",
        text,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    text = text.strip()

    # --------------------------------------------------------
    # DIRECT JSON
    # --------------------------------------------------------

    try:

        parsed = json.loads(
            text
        )

        if not isinstance(
            parsed,
            dict
        ):

            raise ValueError(
                "LLM JSON response is not an object."
            )

        return parsed

    except json.JSONDecodeError:

        pass

    # --------------------------------------------------------
    # TRY TO FIND JSON OBJECT
    # --------------------------------------------------------

    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )

    if start == -1 or end == -1:

        raise ValueError(
            "LLM did not return valid JSON."
        )

    json_text = text[
        start:end + 1
    ]

    try:

        parsed = json.loads(
            json_text
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            "LLM returned invalid JSON."
        ) from exc

    if not isinstance(
        parsed,
        dict
    ):

        raise ValueError(
            "LLM JSON response is not an object."
        )

    return parsed


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

    # --------------------------------------------------------
    # EMPTY CHATQNA
    #
    # IMPORTANT:
    # No LLM call.
    # --------------------------------------------------------

    if not isinstance(
        chatqna,
        str
    ):

        chatqna = ""

    if not chatqna.strip():

        print(
            "\n[CONSENT AGENT] Skipped - empty chatqna."
        )

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

        # ----------------------------------------------------
        # CHAT COMPLETION
        #
        # No with_structured_output().
        # ----------------------------------------------------

        response = client.chat.completions.create(

            model=OPENAI_MODEL,

            temperature=OPENAI_TEMPERATURE,

            messages=[
                {
                    "role": "system",
                    "content":
                        CONSENT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content":
                        f"""
Extract ONLY consent information from
the following chatqna.

CHATQNA
========

{chatqna}

========

Return ONLY valid JSON.
""",
                },
            ],
        )

        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        elapsed = (
            time.perf_counter()
            - start_time
        )

        # ----------------------------------------------------
        # GET MODEL CONTENT
        # ----------------------------------------------------

        if not response.choices:

            raise ValueError(
                "LLM returned no choices."
            )

        message = (
            response.choices[0].message
        )

        content = message.content

        if not content:

            raise ValueError(
                "LLM returned empty content."
            )

        # ----------------------------------------------------
        # PARSE JSON
        # ----------------------------------------------------

        raw_data = extract_json_from_response(
            content
        )

        # ----------------------------------------------------
        # PYDANTIC VALIDATION
        # ----------------------------------------------------

        result = ConsentExtraction.model_validate(
            raw_data
        )

        # ----------------------------------------------------
        # BUILD PERMISSIONS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

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

    print("=" * 70)
    print("CONSENT AGENT - STANDALONE TEST")
    print("=" * 70)

    try:

        with open(
            "input.json",
            "r",
            encoding="utf-8",
        ) as file:

            input_data = json.load(
                file
            )

        chatqna = input_data.get(
            "chatqna",
            ""
        )

        result = consent_agent(
            {
                "chatqna": chatqna
            }
        )

        print(
            "\nCONSENT AGENT OUTPUT:"
        )

        print(
            json.dumps(
                result,
                indent=4,
                ensure_ascii=False,
            )
        )

        # ----------------------------------------------------
        # SAVE STANDALONE OUTPUT
        # ----------------------------------------------------

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
