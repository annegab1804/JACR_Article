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

def resize_with_padding(
    img: Image.Image,
    target_size: tuple[int, int]
) -> Image.Image:
    """Resizes an image while maintaining its original aspect ratio.

    Adds a white background (padding) to ensure the output image exactly
    matches the specified target size without any distortion.

    Args:
        img: The source PIL Image object to be resized.
        target_size: A tuple of two integers (width, height) representing
            the desired dimensions of the output image.

    Returns:
        A new PIL Image object scaled and centered on a white background
        of the target size.
    """
    # Thumbnail scales the image down in-place to fit inside target_size without distortion
    img.thumbnail(target_size, Image.Resampling.LANCZOS)

    # Create a blank white canvas matching the target dimensions
    background = Image.new("RGBA", target_size, "white")

    # Calculate the offsets required to center the scaled image on the canvas
    offset = (
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
    """Combines multiple image panels into a single composite figure.

    Handles 4 panels in a 2x2 grid, 3 panels in a 1x3 horizontal row, and
    2 panels in a 1x2 horizontal row. The size of each panel slot is
    determined by the dimensions of the first valid image in the panel list.
    Missing files will abort the process.

    Args:
        source_dir: The directory path where the source panels are located.
        output_dir: The directory path where the composite figure will be saved.
        output_name: The filename of the final composite image to save.
        panel_names: A list of base filenames (without extensions) to fetch.
        ext: The file extension of the source images. Defaults to ".png".
    """
    images = []
    for name in panel_names:
        path = os.path.join(source_dir, f"{name}{ext}")
        if os.path.exists(path):
            images.append(Image.open(path))
        else:
            print(f"⚠️ Warning: The image {path} is missing. Figure skipped.")
            return

    # Define the target slot size based on the dimensions of the first panel
    target_size = images[0].size
    target_width, target_height = target_size

    # Intelligently resize ALL images (including the first one) to prevent distortion
    resized_images = [
        resize_with_padding(img, target_size) for img in images
    ]

    num_panels: int = len(resized_images)

    # Layout compilation based on panel count
    if num_panels == 4:
        # Arrange in a 2x2 grid
        grid_width = target_width * 2
        grid_height = target_height * 2
        new_img = Image.new("RGBA", (grid_width, grid_height), "white")

        new_img.paste(resized_images[0], (0, 0))
        new_img.paste(resized_images[1], (target_width, 0))
        new_img.paste(resized_images[2], (0, target_height))
        new_img.paste(resized_images[3], (target_width, target_height))

    elif num_panels == 3:
        # Arrange in a single horizontal row (1x3)
        grid_width = target_width * 3
        grid_height = target_height
        new_img = Image.new("RGBA", (grid_width, grid_height), "white")

        new_img.paste(resized_images[0], (0, 0))
        new_img.paste(resized_images[1], (target_width, 0))
        new_img.paste(resized_images[2], (target_width * 2, 0))

    elif num_panels == 2:
        # Arrange in a single horizontal row (1x2)
        grid_width = target_width * 2
        grid_height = target_height
        new_img = Image.new("RGBA", (grid_width, grid_height), "white")

        new_img.paste(resized_images[0], (0, 0))
        new_img.paste(resized_images[1], (target_width, 0))

    elif num_panels == 1:
        # Only 1 panel: use the processed single image directly
        new_img = resized_images[0]

    else:
        print(f"Unsupported layout: {num_panels} panels.")
        return

    # Save the composite layout to the main directory
    output_path = os.path.join(output_dir, output_name)
    new_img.convert("RGB").save(output_path, quality=95)
    print(f"{output_name} successfully created.")
