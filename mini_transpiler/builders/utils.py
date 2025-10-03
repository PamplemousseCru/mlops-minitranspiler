def get_c_list(python_list: list) -> str:
    return "{" + ", ".join(str(x) for x in python_list) + "}"


def get_c_matrix(python_matrix: list) -> str:
    rows = [get_c_list(row) for row in python_matrix]
    return "{" + ", ".join(rows) + "}"


def get_imports() -> str:
    return "#include <stdio.h>\n\n"
