import os
import matplotlib.pyplot as plt

# Fixed series order: actual prices read as neutral ink; model hues are
# CVD-validated categorical slots and must not be reassigned per run.
ACTUAL_COLOR = '#4a4a47'
MODEL_COLORS = ['#2a78d6', '#008300', '#e87ba4', '#eda100']

def plot_predictions(results, ticker, output_path='../outputs/prediction.png'):
    """
    Plots actual test-set prices against each model's predictions.
    `results` is the dict returned by utils.compare_models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    actuals = next(iter(results.values()))['actuals']

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actuals, color=ACTUAL_COLOR, linewidth=1.8, label='Actual')
    for (name, res), color in zip(results.items(), MODEL_COLORS):
        ax.plot(res['predictions'], color=color, linewidth=1.8,
                label=f'{name} (RMSE {res["rmse"]:.2f})')

    ax.set_title(f'{ticker} Close Price — Actual vs Model Predictions (Test Set)')
    ax.set_xlabel('Test set day')
    ax.set_ylabel('Price (USD)')
    ax.grid(color='#e5e4df', linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Prediction plot saved to {output_path}')

def format_metrics_table(results):
    """Formats compare_models results as a markdown table string."""
    lines = ['| Model | MSE | RMSE |', '|-------|-----|------|']
    for name, res in results.items():
        lines.append(f"| {name} | {res['mse']:.4f} | {res['rmse']:.4f} |")
    return '\n'.join(lines)

def save_metrics_report(results, ticker, output_path='../outputs/metrics.md'):
    """Writes the metrics summary table to disk and echoes it to stdout."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    table = format_metrics_table(results)
    report = f'# {ticker} — Model Comparison\n\n{table}\n'
    with open(output_path, 'w') as f:
        f.write(report)
    print(f'\nMetrics summary:\n{table}')
    print(f'Metrics report saved to {output_path}')
