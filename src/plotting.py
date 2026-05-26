import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_reporting_cliff(reporting_rates: dict,
                         ci_dict: dict = None,
                         title: str = 'Demographic reporting cliff',
                         save_path: str = None) -> plt.Figure:
    """Horizontal bar chart of reporting rates, sorted descending (Fig 1A).

    Args:
        reporting_rates: A dictionary mapping labels to reporting rates 
            (e.g., {'Age': 0.82, 'Race': 0.23}).
        ci_dict: An optional dictionary mapping labels to a tuple of confidence 
            intervals (e.g., {'Age': (0.78, 0.86)}). Used to plot error bars.
        title: The title text to display on top of the chart. Defaults to 
            'Demographic reporting cliff'.
        save_path: Filepath where the figure will be saved. If None, the figure 
            is not exported to disk.

    Returns:
        plt.Figure: The generated matplotlib Figure object containing the horizontal 
            bar chart.
    """
    sorted_items = sorted(reporting_rates.items(), key=lambda x: x[1], reverse=True)
    labels = [it[0] for it in sorted_items]
    rates  = [it[1] for it in sorted_items]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.5)))
    bars = ax.barh(labels, rates, color='steelblue', height=0.55)

    if ci_dict:
        xerr_low  = [rates[i] - ci_dict[labels[i]][0] for i in range(len(labels))]
        xerr_high = [ci_dict[labels[i]][1] - rates[i] for i in range(len(labels))]
        ax.errorbar(rates, labels, xerr=[xerr_low, xerr_high],
                    fmt='none', color='black', capsize=4, linewidth=1.2)

    for bar, label, rate in zip(bars, labels, rates):
        text_x_position = ci_dict[label][1] if ci_dict else bar.get_width()
        ax.text(text_x_position + 0.015, bar.get_y() + bar.get_height() / 2,
                f'{rate:.1%}', va='center', fontsize=9)

    ax.set_xlim(0, 1.12)
    ax.set_xlabel('% of devices reporting')
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_lorenz_curve(x: np.ndarray, y: np.ndarray,
                      gini: float = None,
                      title: str = 'Lorenz curve — company HQ concentration',
                      save_path: str = None) -> plt.Figure:
    """Lorenz curve for company HQ concentration (Fig 5B).

    Args:
        x: A 1D NumPy array representing the cumulative share of countries.
        y: A 1D NumPy array representing the cumulative share of devices.
        gini: Optional Gini coefficient to annotate on the chart. If None, 
            the annotation box is omitted.
        title: The title text to display on the chart. Defaults to 
            'Lorenz curve — company HQ concentration'.
        save_path: Filepath where the figure will be saved. If None, the figure 
            is not exported to disk.

    Returns:
        plt.Figure: The generated matplotlib Figure object containing the Lorenz curve plot.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect equality')
    ax.fill_between(x, x, y, alpha=0.25, color='steelblue')
    ax.plot(x, y, color='steelblue', linewidth=2, label='Actual distribution')

    if gini is not None:
        ax.text(0.05, 0.88, f'Gini = {gini:.3f}',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_xlabel('Cumulative share of countries')
    ax.set_ylabel('Cumulative share of devices')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig