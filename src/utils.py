import numpy as np

from train import evaluate_model

def compute_metrics(predictions, actuals):
    """Computes MSE and RMSE between prediction and actual price arrays."""
    mse = float(np.mean((predictions - actuals) ** 2))
    rmse = float(np.sqrt(mse))
    return {'mse': mse, 'rmse': rmse}

def compare_models(models, test_loader, criterion, scaler, device='cpu'):
    """
    Runs each model in `models` (dict of name -> model) through the same test set,
    inverse-scales the outputs back to price space, and computes MSE/RMSE.
    Returns a dict of name -> {'predictions', 'actuals', 'mse', 'rmse'}.
    """
    results = {}
    for name, model in models.items():
        print(f'\nEvaluating {name}...')
        predictions, actuals = evaluate_model(model, test_loader, criterion, device=device)

        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

        metrics = compute_metrics(predictions, actuals)
        results[name] = {'predictions': predictions, 'actuals': actuals, **metrics}
        print(f"{name} — MSE: {metrics['mse']:.4f}, RMSE: {metrics['rmse']:.4f}")

    return results
