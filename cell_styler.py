import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import pathlib
import matplotlib as mpl
import pandas as pd
import numpy as np

class Styler:
    def __init__(self):
        self.set_style()
        self.set_color_palette()
        self.configure_svg_settings()
        
        # Define default figure sizes in points
        self.default_sizes = {
            'single': {'width': 277.5191, 'height': 223.7852},  # Your specified dimensions
            '1.5': {'width': 448.8, 'height': 336.6},
            'double': {'width': 685.0, 'height': 514.0}
        }

        ### Example for creating a figure with specific dimensions ###
        # fig, ax = styler.create_figure(width_pt=250, height_pt=200)
        # fig, ax = styler.create_figure(size_preset='single')

    def set_style(self):
        plt.style.use(['science'])
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 7,
            'axes.linewidth': 0.5,
            'axes.labelsize': 7,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            'lines.linewidth': 0.5,  # Slightly thicker lines
            'lines.markersize': 1,
            # Hide the ticks on the top and right
            'xtick.top': False,
            'ytick.right': False,
            # Make ticks point outward
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            # Disable minor ticks
            'xtick.minor.visible': False,
            'ytick.minor.visible': False,
            'axes.prop_cycle': plt.cycler('color', [
                '#AD5E99',  # Purple
                '#009E73',  # Green
                '#F0E442',  # Yellow
                '#56B4E9',  # Light Blue
                '#E69F00',  # Orange
                '#0072B2',  # Dark Blue
                '#CC79A7',  # Pink
                '#D55E00',  # Dark Orange
                '#000000',  # Black
                '#999999',   # Gray
                '#00743C',   # not color blind green
                '#9B2385'   # not color blind purple 
            ])
        })
        
        # Set spine visibility
        mpl.rc('axes.spines', right=False, top=False)  # Hide top and right spines
        mpl.rc('axes', linewidth=0.5)

    def configure_svg_settings(self):
        """Configure settings specific to SVG text editability"""
        svg_settings = {
            'svg.fonttype': 'none',
            'svg.image_inline': True,
            'svg.hashsalt': None,
            'text.usetex': False,
            'mathtext.fontset': 'dejavusans',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05
        }
        mpl.rcParams.update(svg_settings)

    def set_color_palette(self):
        """Set up color palettes for different plot elements"""
        self.colors = [
            '#AD5E99',  # Purple
            '#009E73',  # Green
            '#F0E442',  # Yellow
            '#56B4E9',  # Light Blue
            '#E69F00',  # Orange
            '#0072B2',  # Dark Blue
            '#CC79A7',  # Pink
            '#D55E00',  # Dark Orange/Red
            '#000000',  # Black
            '#999999',   # Gray
            '#00743C',   # not color blind green
            '#9B2385'   # not color blind purple 
        ]
        
        # Add color mappings for cell lines and drug windows
        self.colors_cell_line = {
            'C57BL6': self.colors[4],  # Orange
            'E14': self.colors[1],     # Green
            'KH2': self.colors[0],     # Purple
        }
        
        self.colors_drug_window = {
            'baseline': self.colors[3],     # Light Blue
            'initial': self.colors[4],      # Orange
            'incubated': self.colors[7],    # Dark Orange/Red
            'burst_peaks': self.colors[0],  # Purple
            'burst': self.colors[9]         # Gray
        }

        self.age_group_colors = {
            '23-33': self.colors[6],  # pink
            '34-45': self.colors[3],  # light blue
            '46-64': self.colors[2]   # yellow
        }

        self.dorsal_ventral_colors = {
            'Dorsal': self.colors[10],  # green
            'Ventral': self.colors[11]   # purple
        }

        self.heatmap_map = {
            'basic_time': 'Blues',
            'intensity': 'viridis'
        }

    def get_color(self, index):
        """Get color by index with fallback to black"""
        if 0 <= index < len(self.colors):
            return self.colors[index]
        return '#000000'  # Default to black

    def get_heatmap_cmap(self, key):
        return self.heatmap_map.get(key, 'viridis')

    def get_cell_figsize(self, columns=1):
        if columns == 1:
            return (3.346, 2.51)
        elif columns == 1.5:
            return (4.488, 3.366)
        elif columns == 2:
            return (6.85, 5.14)
        else:
            raise ValueError("Columns must be 1, 1.5, or 2")
        
    def points_to_inches(self, points):
        """Convert points to inches"""
        return points / 72.0  # Standard conversion: 72 points per inch

    def get_figure_size(self, width_pt=None, height_pt=None, size_preset=None):
        """
        Get figure size in inches. Can specify exact dimensions in points or use presets.
        
        Parameters:
        -----------
        width_pt : float, optional
            Width in points
        height_pt : float, optional
            Height in points
        size_preset : str, optional
            One of 'single', '1.5', 'double' for preset sizes
            
        Returns:
        --------
        tuple : (width in inches, height in inches)
        """
        if width_pt is not None and height_pt is not None:
            return (self.points_to_inches(width_pt), self.points_to_inches(height_pt))
        elif size_preset is not None:
            preset = self.default_sizes.get(size_preset)
            if preset is None:
                raise ValueError(f"Unknown size preset: {size_preset}. Use 'single', '1.5', or 'double'")
            return (self.points_to_inches(preset['width']), self.points_to_inches(preset['height']))
        else:
            # Default to single column size
            preset = self.default_sizes['single']
            return (self.points_to_inches(preset['width']), self.points_to_inches(preset['height']))

    def create_figure(self, nrows=1, ncols=1, width_pt=None, height_pt=None, size_preset=None, **kwargs):
        """
        Create a figure with specified dimensions.
        
        Parameters:
        -----------
        nrows : int
            Number of rows in subplot grid
        ncols : int
            Number of columns in subplot grid
        width_pt : float, optional
            Width in points (as shown in design software)
        height_pt : float, optional
            Height in points
        size_preset : str, optional
            One of 'single', '1.5', 'double' for preset sizes
        **kwargs : 
            Additional arguments passed to plt.subplots
            
        Returns:
        --------
        tuple : (figure, axes)
        """
        figsize = self.get_figure_size(width_pt, height_pt, size_preset)
        fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, **kwargs)
        return fig, ax



    def finish_plot(self, save_plots, save_dir, name, dpi=300):
        if save_plots and save_dir:
            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = pathlib.Path(save_dir)
            
            # Save with specific SVG settings
            svg_kwargs = {
                'format': 'svg',
                'bbox_inches': 'tight',
                'pad_inches': 0.1,
                'transparent': False,
                'dpi': dpi,
                'metadata': {'Creator': 'Matplotlib'}
            }
            
            # Save in different formats
            plt.savefig(save_path / f'{name}.png', dpi=dpi)
            plt.savefig(save_path / f'{name}.svg', **svg_kwargs)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def get_cell_line_color(self, cell_line):
        return self.colors_cell_line.get(cell_line, '#000000')  # Default to black

    def get_drug_window_color(self, window):
        return self.colors_drug_window.get(window, '#000000')  # Default to black
    

    def plot_color_palette(self):
        """
        Create a visual representation of all color palettes available in the Styler.
        This helps in choosing and reviewing colors for different plot elements.
        """
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), height_ratios=[1, 1, 1])
        fig.suptitle('Styler Color Palettes', fontsize=12, y=0.95)
        
        # Plot base colors
        n_colors = len(self.colors)
        for i, color in enumerate(self.colors):
            ax1.bar(i, 1, color=color, width=0.8)
            ax1.text(i, -0.1, f'{i}\n{color}', ha='center', va='top', rotation=0,
                    fontsize=8)
        ax1.set_title('Base Colors')
        ax1.set_xlim(-0.5, n_colors - 0.5)
        ax1.set_ylim(-0.3, 1.2)
        ax1.axis('off')
        
        # Plot cell line colors
        cell_positions = np.arange(len(self.colors_cell_line))
        for i, (cell_line, color) in enumerate(self.colors_cell_line.items()):
            ax2.bar(i, 1, color=color, width=0.8)
            ax2.text(i, -0.1, f'{cell_line}\n{color}', ha='center', va='top',
                    rotation=0, fontsize=8)
        ax2.set_title('Cell Line Colors')
        ax2.set_xlim(-0.5, len(self.colors_cell_line) - 0.5)
        ax2.set_ylim(-0.3, 1.2)
        ax2.axis('off')
        
        # Plot drug window colors
        drug_positions = np.arange(len(self.colors_drug_window))
        for i, (window, color) in enumerate(self.colors_drug_window.items()):
            ax3.bar(i, 1, color=color, width=0.8)
            ax3.text(i, -0.1, f'{window}\n{color}', ha='center', va='top',
                    rotation=0, fontsize=8)
        ax3.set_title('Drug Window Colors')
        ax3.set_xlim(-0.5, len(self.colors_drug_window) - 0.5)
        ax3.set_ylim(-0.3, 1.2)
        ax3.axis('off')
        
        plt.tight_layout()
        plt.show()
        return fig
    
    #  styler.plot_color_palette() can be called to visualize the color palettes
    
def main():
#######for testing standardized plots######
    # Initialize the styler
    styler = Styler()
    
    # Create some example data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = -np.sin(x)
    
    # Create a test plot using the styler's create_figure method
    fig, ax = styler.create_figure(size_preset='single')
    
    # Plot multiple lines to test the color cycle
    ax.plot(x, y1, label='Sin(x)')
    ax.plot(x, y2, label='Cos(x)')
    ax.plot(x, y3, label='-Sin(x)')
    
    # Add labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_title('Test Plot')
    
    # Add legend
    ax.legend()
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()
