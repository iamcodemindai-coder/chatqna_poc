import json
import time
from copy import deepcopy
from typing import Any, Dict, TypedDict

from langgraph.graph import StateGraph, START, END

from agents.supervisor_agent.agent import (
    supervisor_prepare,
    supervisor_finalize,
    supervisor_router,
)

from agents.contact_agent.agent import contact_agent

from agents.consent_agent.agent import consent_agent


# ============================================================
# LANGGRAPH STATE
# ============================================================

class GraphState(TypedDict, total=False):

    # Complete input JSON
    input_data: Dict[str, Any]

    # chatqna extracted by supervisor
    chatqna: str

    # Agent outputs
    contact_result: Dict[str, Any]
    consent_result: Dict[str, Any]

    # Final JSON
    final_output: Dict[str, Any]

    # Status
    status: str
    error: str

    # Timing
    contact_llm_time: float
    consent_llm_time: float
    total_processing_time: float


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph():

    workflow = StateGraph(GraphState)

    # --------------------------------------------------------
    # SUPERVISOR
    # --------------------------------------------------------

    workflow.add_node(
        "supervisor_prepare",
        supervisor_prepare,
    )

    workflow.add_node(
        "supervisor_finalize",
        supervisor_finalize,
    )

    # --------------------------------------------------------
    # AGENTS
    # --------------------------------------------------------

    workflow.add_node(
        "contact_agent",
        contact_agent,
    )

    workflow.add_node(
        "consent_agent",
        consent_agent,
    )

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    workflow.add_edge(
        START,
        "supervisor_prepare",
    )

    # --------------------------------------------------------
    # SUPERVISOR ROUTING
    # --------------------------------------------------------

    workflow.add_conditional_edges(
        "supervisor_prepare",
        supervisor_router,
        {
            "agents": "contact_agent",
            "empty": "supervisor_finalize",
        },
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Contact and Consent are independent.
    #
    # Contact does NOT communicate with Consent.
    # Consent does NOT communicate with Contact.
    #
    # Both operate on the original chatqna.
    # --------------------------------------------------------

    workflow.add_edge(
        "contact_agent",
        "supervisor_finalize",
    )

    workflow.add_edge(
        "consent_agent",
        "supervisor_finalize",
    )

    # --------------------------------------------------------
    # FINAL SUPERVISOR
    # --------------------------------------------------------

    workflow.add_edge(
        "supervisor_finalize",
        END,
    )

    return workflow.compile()


# ============================================================
# PROCESS DATA
# ============================================================

def process_data(
    input_data: Dict[str, Any]
) -> Dict[str, Any]:

    """
    Reusable processing function.

    Current POC:

        input.json
            ↓
        process_data()
            ↓
        output.json

    Future production:

        API JSON
            ↓
        process_data(api_json)
            ↓
        result JSON
    """

    if not isinstance(input_data, dict):

        raise ValueError(
            "Input must be a JSON object."
        )

    graph = build_graph()

    start_time = time.perf_counter()

    initial_state: GraphState = {
        "input_data": deepcopy(input_data),
        "status": "STARTED",
    }

    try:

        result = graph.invoke(
            initial_state
        )

        total_time = (
            time.perf_counter()
            - start_time
        )

        output_data = result.get(
            "final_output"
        )

        if output_data is None:

            output_data = deepcopy(
                input_data
            )

            output_data[
                "processing_status"
            ] = "FAILED"

            output_data[
                "processing_error"
            ] = "Final output was not generated."

        output_data[
            "processing_time_seconds"
        ] = round(
            total_time,
            3,
        )

        return output_data

    except Exception as exc:

        total_time = (
            time.perf_counter()
            - start_time
        )

        print(
            "\n[MAIN] Workflow failed:"
        )

        print(
            str(exc)
        )

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
        ] = "FAILED"

        output_data[
            "processing_error"
        ] = str(exc)

        output_data[
            "processing_time_seconds"
        ] = round(
            total_time,
            3,
        )

        return output_data


# ============================================================
# FILE PROCESSOR
# ============================================================

def process_input_file(
    input_file: str = "input.json",
    output_file: str = "output.json",
):

    print("=" * 70)
    print("CHATQNA LANGGRAPH MULTI-AGENT POC")
    print("=" * 70)

    print(
        f"\nInput : {input_file}"
    )

    print(
        f"Output: {output_file}"
    )

    # --------------------------------------------------------
    # READ INPUT
    # --------------------------------------------------------

    try:

        with open(
            input_file,
            "r",
            encoding="utf-8",
        ) as file:

            input_data = json.load(
                file
            )

    except FileNotFoundError:

        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"Invalid JSON in {input_file}"
        ) from exc

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    output_data = process_data(
        input_data
    )

    # --------------------------------------------------------
    # WRITE OUTPUT
    # --------------------------------------------------------

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output_data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PROCESSING COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "Status:",
        output_data.get(
            "processing_status"
        ),
    )

    print(
        "Contact LLM time:",
        output_data.get(
            "contact_llm_time_seconds",
            0,
        ),
        "seconds",
    )

    print(
        "Consent LLM time:",
        output_data.get(
            "consent_llm_time_seconds",
            0,
        ),
        "seconds",
    )

    print(
        "Total time:",
        output_data.get(
            "processing_time_seconds",
            0,
        ),
        "seconds",
    )

    print(
        "Output file:",
        output_file,
    )

    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_input_file(
        input_file="input.json",
        output_file="output.json",
    )