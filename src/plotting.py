import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

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
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect equality')
    ax.fill_between(x, x, y, alpha=0.25, color='steelblue')
    ax.plot(x, y, color='steelblue', linewidth=2, label='Actual distribution')

    if gini is not None:
        ax.text(0.05, 0.88, f'Gini = {gini:.3f}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='lightgray'))

    ax.set_xlabel('Cumulative share of countries', fontsize=10)
    ax.set_ylabel('Cumulative share of devices', fontsize=10)
    ax.set_title(title, fontsize=10)
    
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(fontsize=10)
    
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

import os
from typing import List, Tuple
from PIL import Image

def paste_centered_on_canvas(
    img: Image.Image,
    target_size: tuple[int, int]
) -> Image.Image:
    """Pastes an image onto a larger white canvas without altering its original zoom.

    Centers the provided image on a clean background matching the maximum cell
    dimensions, which prevents any cropping or aspect ratio distortion.

    Args:
        img (Image.Image): The source PIL Image instance to be processed.
        target_size (Tuple[int, int]): A tuple of two integers (width, height)
            representing the dimensions of the expanded background canvas.

    Returns:
        Image.Image: A new RGBA PIL Image object containing the source image
            centered on a solid white background.
    """
    # Create a white background canvas matching the maximum required dimensions
    background: Image.Image = Image.new("RGBA", target_size, "white")

    # Compute coordinate offsets to center the source image perfectly
    offset: Tuple[int, int] = (
        (target_size[0] - img.size[0]) // 2,
        (target_size[1] - img.size[1]) // 2,
    )
    
    background.paste(img, offset)
    return background


def create_composite_figure(
    source_dir: str,
    output_dir: str,
    output_name: str,
    panel_names: list[str],
    ext: str = ".png"
) -> None:
    """Combines multiple image panels into a uniform composite grid layout.

    Dynamically determines the maximum bounding width and height among all input
    panels. This ensures that smaller elements are padded evenly and larger
    figures are never cropped, preserving the original pixel scale and font sizes.
    Supported structures include 2x2 grids (4 panels), 1x3 rows (3 panels),
    and 1x2 rows (2 panels). Missing source files will abort execution.

    Args:
        source_dir (str): Path to the directory containing the source panels.
        output_dir (str): Path to the directory where the output file will be saved.
        output_name (str): Filename of the final compiled composite image.
        panel_names (List[str]): List of base filenames (without extensions) to fetch.
        ext (str, optional): The file extension of the source panels. Defaults to ".png".

    Returns:
        None
    """
    images: List[Image.Image] = []
    for name in panel_names:
        path: str = os.path.join(source_dir, f"{name}{ext}")
        if os.path.exists(path):
            images.append(Image.open(path))
        else:
            print(f"⚠️ Warning: The image {path} is missing. Figure compilation aborted.")
            return

    # Extract the absolute maximum width and height across ALL panel dimensions
    max_width: int = max(img.size[0] for img in images)
    max_height: int = max(img.size[1] for img in images)
    target_size: Tuple[int, int] = (max_width, max_height)

    # Standardize all panels onto identical non-scaling protective backdrops
    standardized_images: List[Image.Image] = [
        paste_centered_on_canvas(img, target_size) for img in images
    ]

    num_panels: int = len(standardized_images)
    new_img: Image.Image

    # Process grid layout placement mapping based on total panel count
    if num_panels == 4:
        # Assemble panels into a balanced 2x2 matrix
        grid_width: int = max_width * 2
        grid_height: int = max_height * 2
        new_img = Image.new("RGBA", (grid_width, grid_height), "white")
        new_img.paste(standardized_images[0], (0, 0))
        new_img.paste(standardized_images[1], (max_width, 0))
        new_img.paste(standardized_images[2], (0, max_height))
        new_img.paste(standardized_images[3], (max_width, max_height))

    elif num_panels == 3:
        # Assemble panels into a single continuous 1x3 horizontal row
        grid_width = max_width * 3
        grid_height = max_height
        new_img = Image.new("RGBA", (grid_width, grid_height), "white")
        new_img.paste(standardized_images[0], (0, 0))
        new_img.paste(standardized_images[1], (max_width, 0))
        new_img.paste(standardized_images[2], (max_width * 2, 0))

    elif num_panels == 2:
        # Assemble panels into a standard 1x2 horizontal row
        grid_width = max_width * 2
        grid_height = max_height
        new_img = Image.new("RGBA", (grid_width, grid_height), "white")
        new_img.paste(standardized_images[0], (0, 0))
        new_img.paste(standardized_images[1], (max_width, 0))

    elif num_panels == 1:
        # Bypass canvas layout assembly for isolated single panel inputs
        new_img = standardized_images[0]
        
    else:
        print(f"Unsupported layout configuration: {num_panels} panels.")
        return

    # Export final figure with unified spatial alignments
    output_path: str = os.path.join(output_dir, output_name)
    new_img.convert("RGB").save(output_path, quality=95)
    print(f"{output_name} successfully created!")