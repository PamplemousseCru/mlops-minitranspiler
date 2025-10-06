from builders.utils import get_c_list, get_c_matrix, get_imports
from functions.utils import linear_regression
from sklearn.tree import DecisionTreeClassifier


def _main(features, thresholds, children_left, children_right, values, n_nodes, input, n_input, classes, n_classes) -> str:
    return f"""
int main(int argc, char** argv) {{
    int features[{n_nodes}] = {features};
    double thresholds[{n_nodes}] = {thresholds};
    int children_left[{n_nodes}] = {children_left};
    int children_right[{n_nodes}] = {children_right};
    double values[{n_nodes}][2] = {values};
    double input[{n_input}] = {input};
    char* classes[{n_classes}] = {classes};

    int current_node = 0;
    while (features[current_node] != -2) {{
        if (input[features[current_node]] <= thresholds[current_node]) {{
            current_node = children_left[current_node];
        }}
        else {{
            current_node = children_right[current_node];
        }}
    }}

    int class_index = 0;
    double best_value = values[current_node][0];
    for (int i = 1; i < {n_classes}; i++) {{
        if (values[current_node][i] > best_value) {{
            best_value = values[current_node][i];
            class_index = i;
        }}
    }}

    printf("Predicting class \\"%s\\"\\n", classes[class_index]);
    return 0;
}}
"""

def sanitize_values(model: DecisionTreeClassifier):
    result = []
    for value_mat in model.tree_.value:
        result.append(list(value_mat[0]))
    return result

def decision_tree_builder(model: DecisionTreeClassifier, input) -> str:
    features = get_c_list(model.tree_.feature)
    thresholds = get_c_list(model.tree_.threshold)
    children_left = get_c_list(model.tree_.children_left)
    children_right = get_c_list(model.tree_.children_right)
    values = get_c_matrix(sanitize_values(model))

    n_input = len(input)
    input = get_c_list(input)
    
    n_nodes = len(model.tree_.feature)
    n_classes = len(model.classes_)
    classes = get_c_list(map(lambda x : f"\"{str(x)}\"", model.classes_)) # to make sure we get strings

    return f"""{get_imports()}
{_main(features, thresholds, children_left, children_right, values, n_nodes, input, n_input, classes, n_classes)}
"""
