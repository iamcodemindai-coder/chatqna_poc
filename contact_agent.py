import json
import os
import re
import time
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict


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
# Databricks Model Serving exposes an OpenAI-compatible
# Chat Completions interface.
# ============================================================

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


# ============================================================
# CONTACT SCHEMA
# ============================================================

class ContactPerson(BaseModel):

    model_config = ConfigDict(
        extra="ignore"
    )

    Type: str = Field(
        description=(
            "Must be HCP or Patient."
        )
    )

    FirstName: Optional[str] = None

    LastName: Optional[str] = None

    PhoneNumber: Optional[str] = None

    AdditionalPhoneNumber: Optional[str] = None

    EmailAddress: Optional[str] = None

    DateOfBirth: Optional[str] = None

    Gender: Optional[str] = None

    Address: Optional[str] = None

    Street: Optional[str] = None

    City: Optional[str] = None

    ZipPostalCode: Optional[str] = None

    StateProvince: Optional[str] = None

    Country: Optional[str] = None


class ContactExtraction(BaseModel):

    model_config = ConfigDict(
        extra="ignore"
    )

    contacts: List[
        ContactPerson
    ] = Field(
        default_factory=list
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

CONTACT_SYSTEM_PROMPT = """

You are the CONTACT AGENT.

Your ONLY responsibility is to extract CONTACT information
from the provided chatqna.

Do NOT extract consent or permissions.

============================================================
CONTACT TYPES
============================================================

Allowed values:

HCP
Patient

NEVER return:

Both

If both HCP and Patient are present, return two separate
objects.

Maximum:

one HCP
one Patient

============================================================
CONTACT FIELDS
============================================================

Extract:

Type
FirstName
LastName
PhoneNumber
AdditionalPhoneNumber
EmailAddress
DateOfBirth
Gender
Address
Street
City
ZipPostalCode
StateProvince
Country

============================================================
NOISY DATA HANDLING
============================================================

The chatqna can contain:

- unwanted characters
- special symbols
- HTML
- broken text
- transcription errors
- spelling mistakes
- repeated text
- irrelevant text
- different languages
- mixed languages
- conversational filler

Understand the meaningful semantic content.

Ignore irrelevant noise.

Do NOT remove meaningful information simply because the
surrounding text is noisy.

Understand contact information even when it is written in
different languages.

============================================================
CONTACT IDENTIFICATION
============================================================

Examples:

"I am a doctor"
"I am a physician"
"I am an HCP"

=> HCP

"I am the patient"
"I am calling as the patient"

=> Patient

"The patient's name is Mary"

=> Patient

"The doctor is John Smith"

=> HCP

Always determine which person the information belongs to.

Never move HCP information into Patient fields.

Never move Patient information into HCP fields.

============================================================
IMPORTANT RULES
============================================================

1. NEVER invent information.

2. Missing fields must be null.

3. Preserve phone numbers accurately.

4. Preserve email addresses accurately.

5. Normalize an unambiguous date of birth to YYYY-MM-DD.

6. Preserve gender exactly as reliably stated.

7. Do not guess address components.

8. If an address is combined, only populate components that
   can be reliably identified.

9. Prefer clearly confirmed information.

10. If conflicting information exists:
    - prefer customer-confirmed information
    - otherwise prefer latest clearly stated value
    - never invent a resolution

11. Ignore greetings and small talk.

12. Ignore unrelated case/product information.

13. Do not extract from unrelated JSON fields.

14. NEVER return Type=Both.

15. Maximum one HCP and one Patient.

16. Return ONLY valid JSON.

17. Do not return Markdown.

18. Do not return ```json.

19. Do not return explanations.

============================================================
OUTPUT FORMAT
============================================================

Return exactly this structure:

{
    "contacts": [
        {
            "Type": "HCP",
            "FirstName": null,
            "LastName": null,
            "PhoneNumber": null,
            "AdditionalPhoneNumber": null,
            "EmailAddress": null,
            "DateOfBirth": null,
            "Gender": null,
            "Address": null,
            "Street": null,
            "City": null,
            "ZipPostalCode": null,
            "StateProvince": null,
            "Country": null
        }
    ]
}

If there are no contacts:

{
    "contacts": []
}

"""


# ============================================================
# CONTACT FIELDS
# ============================================================

CONTACT_FIELDS = [
    "FirstName",
    "LastName",
    "PhoneNumber",
    "AdditionalPhoneNumber",
    "EmailAddress",
    "DateOfBirth",
    "Gender",
    "Address",
    "Street",
    "City",
    "ZipPostalCode",
    "StateProvince",
    "Country",
]


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
# NORMALIZE CONTACT
# ============================================================

def normalize_contact(
    contact: Dict[str, Any]
):

    contact_type = str(
        contact.get(
            "Type",
            ""
        )
    ).strip()

    # --------------------------------------------------------
    # Validate Type
    # --------------------------------------------------------

    if contact_type not in [
        "HCP",
        "Patient",
    ]:

        return None

    normalized = {
        "Type": contact_type
    }

    # --------------------------------------------------------
    # Normalize Fields
    # --------------------------------------------------------

    for field in CONTACT_FIELDS:

        value = contact.get(
            field
        )

        if value is None:

            value = ""

        normalized[field] = str(
            value
        ).strip()

    return normalized


# ============================================================
# CONTACT AGENT
# ============================================================

def contact_agent(
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
            "\n[CONTACT AGENT] Skipped - empty chatqna."
        )

        return {
            "contact_result": {
                "contacts": []
            },
            "contact_llm_time": 0.0,
        }

    print(
        "\n[CONTACT AGENT] Running LLM..."
    )

    start_time = time.perf_counter()

    try:

        # ----------------------------------------------------
        # CHAT COMPLETION
        #
        # No with_structured_output().
        #
        # This avoids LangChain adding provider-specific
        # structured output/tool parameters.
        # ----------------------------------------------------

        response = client.chat.completions.create(

            model=OPENAI_MODEL,

            temperature=OPENAI_TEMPERATURE,

            messages=[
                {
                    "role": "system",
                    "content":
                        CONTACT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content":
                        f"""
Extract contact information from
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

        validated = ContactExtraction.model_validate(
            raw_data
        )

        # ----------------------------------------------------
        # NORMALIZE CONTACTS
        # ----------------------------------------------------

        contacts = []

        seen_types = set()

        for contact_model in validated.contacts:

            normalized = normalize_contact(
                contact_model.model_dump()
            )

            if normalized is None:

                continue

            contact_type = normalized[
                "Type"
            ]

            # ------------------------------------------------
            # Only one HCP and one Patient
            # ------------------------------------------------

            if contact_type in seen_types:

                continue

            seen_types.add(
                contact_type
            )

            contacts.append(
                normalized
            )

        # ----------------------------------------------------
        # DETERMINISTIC ORDER
        # ----------------------------------------------------

        contacts.sort(
            key=lambda x:
            0 if x["Type"] == "HCP"
            else 1
        )

        print(
            "[CONTACT AGENT] Completed."
        )

        print(
            "[CONTACT AGENT] Time:",
            round(
                elapsed,
                3
            ),
            "seconds",
        )

        print(
            "[CONTACT AGENT] Contacts:",
            len(contacts),
        )

        return {

            "contact_result": {
                "contacts": contacts
            },

            "contact_llm_time": round(
                elapsed,
                3
            ),
        }

    except Exception as exc:

        print(
            "[CONTACT AGENT] ERROR:",
            str(exc)
        )

        raise RuntimeError(
            f"Contact Agent failed: {exc}"
        ) from exc


# ============================================================
# STANDALONE TEST RUNNER
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CONTACT AGENT - STANDALONE TEST")
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

        result = contact_agent(
            {
                "chatqna": chatqna
            }
        )

        print(
            "\nCONTACT AGENT OUTPUT:"
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
            "contact_output.json",
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
            "\nContact agent output saved to: "
            "contact_output.json"
        )

    except Exception as exc:

        print(
            "\nContact agent test failed:"
        )

        print(
            str(exc)
        )
