from copy import deepcopy
from typing import Any, Dict, List


# ============================================================
# CONSTANTS
# ============================================================

CHATQNA_FIELDS = [
    "chatqna",
    "ChatQnA",
    "ChatQNA",
    "ChatQandA",
    "ChatQANDA",
]


PERMISSION_FIELDS = [
    "Permission To Contact HCP",
    "Permission To Contact Reporter",
    "Permission To Contact Patient",
    "Permission To Contact Complaint",
    "Permission To Contact Via Text",
]


# ============================================================
# FIND CHATQNA
# ============================================================

def find_chatqna(
    input_data: Dict[str, Any]
):

    for field in CHATQNA_FIELDS:

        if field in input_data:

            value = input_data.get(
                field
            )

            if isinstance(
                value,
                str
            ):

                return value

    return None


# ============================================================
# SUPERVISOR PREPARE
# ============================================================

def supervisor_prepare(
    state: Dict[str, Any]
):

    input_data = state.get(
        "input_data",
        {},
    )

    print(
        "\n[SUPERVISOR] Input received."
    )

    chatqna = find_chatqna(
        input_data
    )

    # --------------------------------------------------------
    # CHATQNA VALIDATION
    # --------------------------------------------------------

    if (
        chatqna is None
        or not chatqna.strip()
    ):

        print(
            "[SUPERVISOR] chatqna is empty."
        )

        return {
            "chatqna": "",
            "status": "SKIPPED_EMPTY_CHATQNA",
        }

    print(
        "[SUPERVISOR] chatqna found."
    )

    print(
        "[SUPERVISOR] Routing to agents."
    )

    return {
        "chatqna": chatqna,
        "status": "READY",
    }


# ============================================================
# SUPERVISOR ROUTER
# ============================================================

def supervisor_router(
    state: Dict[str, Any]
):

    if state.get(
        "status"
    ) == "SKIPPED_EMPTY_CHATQNA":

        return "empty"

    return "agents"


# ============================================================
# PERMISSION CONVERSION
# ============================================================

def permission_to_output(
    value
):

    if value is True:
        return "Yes"

    if value is False:
        return "No"

    return ""


# ============================================================
# BUILD CHAT RESPONSE
# ============================================================

def build_chat_response(
    contacts: List[Dict[str, Any]]
):

    return [
        {
            "Contact": contact
        }
        for contact in contacts
    ]


# ============================================================
# BUILD ADDITIONAL FIELDS
# ============================================================

def build_additional_fields(
    permissions: Dict[str, Any]
):

    return [

        {
            "Permission To Contact HCP":
                permission_to_output(
                    permissions.get(
                        "Permission To Contact HCP"
                    )
                )
        },

        {
            "Permission To Contact Reporter":
                permission_to_output(
                    permissions.get(
                        "Permission To Contact Reporter"
                    )
                )
        },

        {
            "Permission To Contact Patient":
                permission_to_output(
                    permissions.get(
                        "Permission To Contact Patient"
                    )
                )
        },

        {
            "Permission To Contact Complaint":
                permission_to_output(
                    permissions.get(
                        "Permission To Contact Complaint"
                    )
                )
        },

        {
            "Permission To Contact Via Text":
                permission_to_output(
                    permissions.get(
                        "Permission To Contact Via Text"
                    )
                ),
        },
    ]


# ============================================================
# EMPTY OUTPUT
# ============================================================

def build_empty_output(
    input_data: Dict[str, Any]
):

    output_data = deepcopy(
        input_data
    )

    output_data[
        "ChatResponse"
    ] = []

    output_data[
        "Additional Fields"
    ] = [

        {
            "Permission To Contact HCP": ""
        },

        {
            "Permission To Contact Reporter": ""
        },

        {
            "Permission To Contact Patient": ""
        },

        {
            "Permission To Contact Complaint": ""
        },

        {
            "Permission To Contact Via Text": ""
        },
    ]

    output_data[
        "processing_status"
    ] = "SKIPPED_EMPTY_CHATQNA"

    output_data[
        "contact_llm_time_seconds"
    ] = 0.0

    output_data[
        "consent_llm_time_seconds"
    ] = 0.0

    return output_data


# ============================================================
# SUPERVISOR FINALIZE
# ============================================================

def supervisor_finalize(
    state: Dict[str, Any]
):

    input_data = state.get(
        "input_data",
        {},
    )

    # --------------------------------------------------------
    # EMPTY CHATQNA
    # --------------------------------------------------------

    if state.get(
        "status"
    ) == "SKIPPED_EMPTY_CHATQNA":

        print(
            "[SUPERVISOR] No agent call required."
        )

        return {
            "final_output":
                build_empty_output(
                    input_data
                )
        }

    # --------------------------------------------------------
    # CONTACT RESULT
    # --------------------------------------------------------

    contact_result = state.get(
        "contact_result",
        {
            "contacts": []
        },
    )

    contacts = contact_result.get(
        "contacts",
        [],
    )

    # --------------------------------------------------------
    # CONSENT RESULT
    # --------------------------------------------------------

    consent_result = state.get(
        "consent_result",
        {
            "permissions": {}
        },
    )

    permissions = consent_result.get(
        "permissions",
        {},
    )

    # --------------------------------------------------------
    # BUILD OUTPUT
    # --------------------------------------------------------

    output_data = deepcopy(
        input_data
    )

    output_data[
        "ChatResponse"
    ] = build_chat_response(
        contacts
    )

    output_data[
        "Additional Fields"
    ] = build_additional_fields(
        permissions
    )

    output_data[
        "processing_status"
    ] = "SUCCESS"

    output_data[
        "contact_llm_time_seconds"
    ] = state.get(
        "contact_llm_time",
        0.0,
    )

    output_data[
        "consent_llm_time_seconds"
    ] = state.get(
        "consent_llm_time",
        0.0,
    )

    print(
        "\n[SUPERVISOR] Final output generated."
    )

    print(
        "[SUPERVISOR] Contacts:",
        len(contacts),
    )

    return {
        "final_output": output_data
    }