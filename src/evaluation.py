import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


def plot_results(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, max(y_true)], [0, max(y_true)], "r--")
    plt.xlabel("Actual Defect Percentage")
    plt.ylabel("Predicted Defect Percentage")
    plt.title("Actual vs Predicted Defect Percentage")
    plt.grid(True)
    plt.savefig("../data/outputs/regression_results.png")
    plt.close()


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    plot_results(y_test, y_pred)
    return {"mae": mae, "r2": r2}
