import joblib
import sys

def inspect():
    with open('c:/tmp/model_info.txt', 'a') as out:
        try:
            model = joblib.load('c:/Users/Lenovo/Downloads/skincare_ai/back-end/model.pkl')
            if hasattr(model, 'feature_names_in_'):
                out.write(f"Feature Names: {model.feature_names_in_}\n")
            if hasattr(model, 'n_features_in_'):
                out.write(f"Num Features: {model.n_features_in_}\n")
            if hasattr(model, 'n_outputs_'):
                out.write(f"Num Outputs: {model.n_outputs_}\n")
        except Exception as e:
            out.write(f"Error: {e}\n")

if __name__ == '__main__':
    inspect()
