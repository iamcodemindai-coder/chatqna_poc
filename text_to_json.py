import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TXT_DIR = BASE_DIR / "txt_data"
JSON_DIR = BASE_DIR / "json_data"


# ============================================================
# TXT -> JSON
# ============================================================

def convert_txt_to_json(
    txt_file: Path,
    json_file: Path,
):
    """
    Convert one TXT transcript into JSON.

    The complete TXT content is stored inside
    the 'chatqna' field.
    """

    # --------------------------------------------------------
    # READ TXT
    # --------------------------------------------------------

    with open(
        txt_file,
        "r",
        encoding="utf-8",
    ) as file:

        chatqna = file.read()

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    if not chatqna.strip():

        print(
            f"[SKIPPED] Empty file: {txt_file.name}"
        )

        return False

    # --------------------------------------------------------
    # CASE ID
    #
    # CASE-001.txt
    #       ↓
    # caseId = CASE-001
    # --------------------------------------------------------

    case_id = txt_file.stem

    # --------------------------------------------------------
    # BUILD JSON
    # --------------------------------------------------------

    output_data = {

        "caseId": case_id,

        "source": "frontend",

        "chatqna": chatqna,
    }

    # --------------------------------------------------------
    # WRITE JSON
    # --------------------------------------------------------

    with open(
        json_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output_data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"[CONVERTED] {txt_file.name}"
        f" -> {json_file.name}"
    )

    return True


# ============================================================
# BATCH CONVERTER
# ============================================================

def batch_convert():

    print("=" * 70)
    print("TXT -> JSON BATCH CONVERTER")
    print("=" * 70)

    print(
        f"\nInput folder : {TXT_DIR}"
    )

    print(
        f"Output folder: {JSON_DIR}"
    )

    # --------------------------------------------------------
    # CREATE FOLDERS IF NOT EXISTS
    # --------------------------------------------------------

    TXT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    JSON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # FIND ALL TXT FILES
    # --------------------------------------------------------

    txt_files = sorted(
        TXT_DIR.glob("*.txt")
    )

    # --------------------------------------------------------
    # NO TXT FILE
    # --------------------------------------------------------

    if not txt_files:

        print(
            "\n[INFO] No .txt files found."
        )

        print(
            f"Put TXT files inside:"
            f"\n{TXT_DIR}"
        )

        return

    # --------------------------------------------------------
    # PROCESS ALL FILES
    # --------------------------------------------------------

    converted = 0
    skipped = 0

    for txt_file in txt_files:

        json_file = (
            JSON_DIR
            / f"{txt_file.stem}.json"
        )

        try:

            success = convert_txt_to_json(
                txt_file,
                json_file,
            )

            if success:

                converted += 1

            else:

                skipped += 1

        except Exception as exc:

            skipped += 1

            print(
                f"[ERROR] {txt_file.name}: "
                f"{exc}"
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CONVERSION COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Total TXT files : {len(txt_files)}"
    )

    print(
        f"Converted        : {converted}"
    )

    print(
        f"Skipped/Failed   : {skipped}"
    )

    print(
        f"JSON location    : {JSON_DIR}"
    )

    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    batch_convert()
