from builders.utils import get_c_list, get_c_matrix, get_imports, get_malloc
from functions.utils import linear_regression, get_exp_approx, get_sigmoid, get_softmax
from sklearn.linear_model import LogisticRegression


def _main_n_classes(features, thetas, classes, n_thetas, n_features, n_classes) -> str:
    return f"""
int main(int argc, char** argv) {{
    char* classes[{n_classes}] = {classes};
    double features[{n_features}] = {features};
    double thetas[{n_classes}][{n_thetas}] = {thetas};

    double all_predictions[{n_classes}];

    int best_index = 0;
    double best_prediction = linear_regression_prediction(features, thetas[0], {n_thetas});
    all_predictions[0] = best_prediction;
    

    for (int i = 1; i < {n_classes}; i++) {{
        double pred = linear_regression_prediction(features, thetas[i], {n_thetas});
        all_predictions[i] = pred;
        if (pred > best_prediction) {{
            best_index = i;
            best_prediction = pred;
        }}
    }}
    
    double* softmaxed_preds = softmax(all_predictions, {n_classes});
    printf("Predicting class \\"%s\\" with probability %f\\n", classes[best_index], softmaxed_preds[best_index]);
    free(softmaxed_preds);
    return 0;
}}
"""

def _main_2_classes(features, thetas, classes, n_thetas, n_features) -> str:
    return f"""
int main(int argc, char** argv) {{
    char* classes[2] = {classes};
    double features[{n_features}] = {features};
    double thetas[{n_thetas}] = {thetas};

    double pred_class_1 = linear_regression_prediction(features, thetas, {n_thetas});
    double proba_class_1 = sigmoid(pred_class_1);
    
    int class_index = 1;
    double class_proba = proba_class_1;

    if (proba_class_1 < 0.5) {{
        class_index = 0;
        class_proba = 1 - proba_class_1;
    }}

    printf("Predicting class \\"%s\\" with probability %f\\n", classes[class_index], class_proba);
    return 0;
}}
"""

def logistic_regression_builder(model: LogisticRegression, features) -> str:
    if len(model.coef_) == 1:
        n_thetas = len(model.coef_[0]) + 1 # add intercept
        n_features = len(features)
        
        features = get_c_list(features)
        thetas = get_c_list(model.intercept_.tolist() + model.coef_[0].tolist())
        classes = get_c_list(map(lambda x : f"\"{str(x)}\"", model.classes_)) # just to make sure we get strings

        return f"""{get_imports()}
{get_exp_approx()}
{get_sigmoid()}
{linear_regression()}
{_main_2_classes(features, thetas, classes, n_thetas, n_features)}
"""
    else:
        height, n_thetas = model.coef_.shape
        n_thetas += 1 # add intercept
        n_features = len(features)
        n_classes = len(model.classes_)

        features = get_c_list(features)
        coefs = model.coef_.tolist()
        for i, intercept in enumerate(model.intercept_):
            coefs[i] = [intercept] + coefs[i]
        thetas = get_c_matrix(coefs)
        classes = get_c_list(map(lambda x : f"\"{str(x)}\"", model.classes_)) # just to make sure we get strings
        
        return f"""{get_imports()}
{get_malloc()}
{get_exp_approx()}
{get_softmax()}
{linear_regression()}
{_main_n_classes(features, thetas, classes, n_thetas, n_features, n_classes)}
"""
