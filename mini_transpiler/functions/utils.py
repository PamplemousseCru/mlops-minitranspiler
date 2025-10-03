def get_exp_approx() -> str:
    return (
        "double exp_approx(double x, int n) {\n"
        "    double result = 1.0;\n"
        "    double term = 1.0;\n"
        "    for (int i = 1; i <= n; i++) {\n"
        "        term *= x / i;\n"
        "        result += term;\n"
        "    }\n"
        "    return result;\n"
        "}\n\n"
    )


def get_sigmoid_inline(var_name: str) -> str:
    return f"1 / (1 + exp_approx(-{var_name}, 10))"


def get_sigmoid() -> str:
    return "double sigmoid(double x) {\n    return 1 / (1 + exp_approx(-x, 10));\n}\n\n"


def linear_regression() -> str:
    return """
double linear_regression_prediction(double* features, double* thetas, int n_parameters) {{
    double result = thetas[0];
    for (int i = 0; i < n_parameters - 1; i++) {{
        result += features[i] * thetas[i + 1];
    }}
    return result;
}}
"""
