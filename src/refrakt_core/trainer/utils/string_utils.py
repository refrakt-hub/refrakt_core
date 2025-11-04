import re


def to_snake_case(name):
    if name.isupper() or name.islower():
        return name.lower()
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    name = name.replace("__", "_")
    return name.strip("_")
