# How to run the code:
- run "python tain_models.py", it will train 4 models and dump them as .joblib files
- run "python mini_transpiler/main.py", it will transpile the 4 models to their respective c files, as well as tell you what the predictions were in Python, so you can compare with the outputs of the C files
- compile and run each C file to check the results

# PS:
The 4 models are:
- A linear regression
- A decision tree
- 2 logistic regression (one with 2 classes, and one with 3, because the models behave differently when having n or 2 classes)