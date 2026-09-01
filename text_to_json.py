import json


def convert_txt_to_json(
    txt_file="chatqna.txt",
    json_file="input.json"
):

    with open(
        txt_file,
        "r",
        encoding="utf-8"
    ) as file:

        transcript = file.read()

    data = {
        "caseId": "CASE-001",
        "source": "frontend",
        "someExistingField": "existing value",
        "chatqna": transcript
    }

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("Conversion completed.")
    print(
        "Characters:",
        len(transcript)
    )


if __name__ == "__main__":

    convert_txt_to_json()