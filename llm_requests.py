import json
import requests


def llm_request_correct_names_with_age(children):
    children_text = "\n".join(f"- {c['name']} (age {c['age']})" for c in children)

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3.1:8b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You clean OCR results of children (name, age). "
                    "Always return names with FULL names in 'Surname, Givenname' format, all are Dutch names"
                    "Names should be proper capitalization, spacing, commas, dots, and accents. "
                    # "Enties with different age are never duplicated entries "
                    # "For similar entries the difference may be only spaces, comma or dot, or one or two letters. "
                    # "Set entries as duplicated only when first name and last name both similar. "
                    # "Remove duplicate entries. "
                    "Do not change the order of entries. "
                    'In the output array, each object must have exactly: {"name", "age"}'
                    "Output ONLY a JSON array of strings, nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Reference names:\n{children_text}\n\n"
                    "Return the corrected name list as JSON only."
                ),
            },
        ],
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}

    structured_data = []
    try:
        response = requests.post(url, headers=headers, json=payload)
        raw = response.json()
        print(raw)
        content = raw["message"]["content"].strip()

        # Sometimes model adds text like "Here is the JSON:", try extracting JSON only
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end != -1:
            content = content[start:end]
        else:
            content = f"[{content}]"

        structured_data = json.loads(content)
        return structured_data

    except Exception as e:
        print(f"Error request: {e}")
        return children


def llm_request_correct_names(names):
    name_text = "\n".join(f"- {name}" for name in names)

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3.1:8b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You clean OCR results of children's names. "
                    "Always return FULL names in 'Surname, Givenname' format, "
                    "Do not change names order in the current list "
                    "with proper capitalization, spacing, commas, dots, and accents. "
                    "Output ONLY a JSON array of strings, nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Reference names:\n{name_text}\n\n"
                    "Return the corrected name list as JSON only."
                ),
            },
        ],
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}

    structured_data = []
    try:
        response = requests.post(url, headers=headers, json=payload)
        raw = response.json()
        print(raw)
        content = raw["message"]["content"].strip()

        # Sometimes model adds text like "Here is the JSON:", try extracting JSON only
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end != -1:
            content = content[start:end]
        else:
            content = f"[{content}]"

        structured_data = json.loads(content)
        return structured_data

    except Exception as e:
        print(f"Error request: {e}")
        return names


def llm_request_child_planning(def_names, children):
    """
    Correct children's names using LLM with reference def_names.
    - Keep order and number of children unchanged.
    - Match names to def_names if spelling is close AND age matches.
    - Otherwise, just normalize the name (capitalization, commas, dots, spacing).
    - Do not replace with unrelated def_names.
    - Return ONLY JSON array of children objects [{name, age}, ...].
    """

    # Prepare def_names list
    name_text = "\n".join(f"- {name}" for name in def_names)

    # Prepare children list text
    children_text = "\n".join(f"- {c['name']} (age {c['age']})" for c in children)

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3.1:8b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a name normalization assistant.\n\n"
                    "TASK:\n"
                    "Fix children List's names using a reference list of correct names when they are similar.\n\n"
                    "RULES:\n"
                    "1. Do NOT add, remove, or reorder children list. Keep the exact same number and order. So the result count and order should be same as original children list.\n"
                    "2. If a child's name nearly matches one in the reference list, replace it with the exact reference name.(Remember: The difference is only one or two letters, comma or dot, spaces or capital letters. Do not affect when the difference of name of child and reference list is much than above.)\n"
                    "3. keep original children list, only change items when you are sure it's name is almost similar to one in reference list.\n"
                    "4. Do NOT replace with unrelated names from children list.\n"
                    "5. Output ONLY valid JSON array of objects, no extra text.\n"
                    '6. Each object must have exactly: {"name", "age"}.\n'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Reference names:\n{name_text}\n\n"
                    f"Children List to correct:\n{children_text}\n\n"
                    "Return the corrected children list as JSON only."
                ),
            },
        ],
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}

    structured_data = []
    try:
        response = requests.post(url, headers=headers, json=payload)
        raw = response.json()
        print(raw)
        content = raw["message"]["content"].strip()

        # Sometimes model adds text like "Here is the JSON:", try extracting JSON only
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end != -1:
            content = content[start:end]
        else:
            content = f"[{content}]"

        structured_data = json.loads(content)

    except Exception as e:
        print(f"Error request: {e}")
        return children

    return structured_data
