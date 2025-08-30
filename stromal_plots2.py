# filepath: /Volumes/hunter_ssd/endometrial_scRNA/analysis/tn_2022/stromal_plots.py

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
import os
from scipy.stats import mannwhitneyu
from IPython import embed
from cell_styler import Styler

styler = Styler()

# Set up plotting parameters for publication quality
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 18,  # 18pt for titles
    'axes.labelsize': 18,  # 18pt for x/y axis labels
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,  # 14pt for legends
    'figure.titlesize': 18,  # 18pt for figure titles
    'font.family': ['Helvetica', 'Arial']  # Try Helvetica first, fallback to Arial
})

# Load data and set up directories
stromal_path = '/Volumes/hunter_ssd/endometrial_scRNA/analysis/tn_2022/endo-2022_stromal.h5ad'
stromal = ad.read_h5ad(stromal_path)
plot_dir = '/Volumes/hunter_ssd/endometrial_scRNA/analysis/tn_2022/plots2/'
os.makedirs(plot_dir, exist_ok=True)

# Filter to Ctrl and EuE samples
ctrl_eue_mask = stromal.obs['sample_type_rename'].isin(['Ctrl', 'EuE'])
stromal_filtered = stromal[ctrl_eue_mask, :].copy()

print(f"Filtered data: {stromal_filtered.n_obs} cells (Ctrl and EuE only)")

# Find sample ID column
sample_id_col = None
for col in ['sample_id', 'patient_id', 'PID', 'sample', 'patient', 'donor_id']:
    if col in stromal_filtered.obs.columns:
        sample_id_col = col
        print(f"Using sample ID column: {col}")
        break

if not sample_id_col:
    print("Warning: No sample ID column found. Cannot perform pseudobulk analysis.")

# Define genes and color scheme
genes = ['NBN', 'RAD50', 'MRE11', 'RPA2', 'TP53BP1']
# Color scheme: Control = white, Endometriosis = gray
condition_colors = {'Ctrl': '#FFFFFF', 'EuE': '#808080'}

def calculate_pseudobulk_stats(data, gene, sample_id_col):
    """Calculate pseudobulk statistics for a gene"""
    pseudobulk_data = []
    
    for condition in ['Ctrl', 'EuE']:
        condition_data = data[data.obs['sample_type_rename'] == condition]
        for sample in condition_data.obs[sample_id_col].unique():
            sample_data = condition_data[condition_data.obs[sample_id_col] == sample]
            if sample_data.n_obs > 0:
                sample_expr = sample_data[:, gene].X
                if hasattr(sample_expr, 'toarray'):
                    sample_expr = sample_expr.toarray().flatten()
                else:
                    sample_expr = sample_expr.flatten()
                
                mean_expr = np.mean(sample_expr)
                pseudobulk_data.append({
                    'sample_id': sample,
                    'condition': condition,
                    'mean_expression': mean_expr,
                    'n_cells': sample_data.n_obs
                })
    
    return pd.DataFrame(pseudobulk_data)

def add_significance_annotation(ax, x1, x2, y, p_value, height_offset=0.5):
    """Add significance annotation to plot matching RPA2 style"""
    # Determine significance level
    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    elif p_value < 0.1:
        sig_text = '†'
    else:
        sig_text = 'ns'
    
    # Add horizontal line only (clean style like RPA2)
    line_height = y + height_offset
    ax.plot([x1, x2], [line_height, line_height], 'k-', linewidth=1.5)
    ax.text((x1 + x2) / 2, line_height + 0.2, sig_text, 
            ha='center', va='bottom', fontsize=20, fontweight='bold')

def create_publication_plot(data, gene, pseudobulk_df, save_path_base, font_family='Helvetica', y_max=None):
    """Create publication-ready plot with RPA2 style"""
    # Set font family for this plot
    plt.rcParams['font.family'] = font_family
    
    # Use styler to create figure with SVG-friendly settings
    fig, ax = styler.create_figure(width_pt=432, height_pt=540)  # Taller to match RPA2
    
    # Get gene expression data for individual cells
    gene_expr = data[:, gene].X
    if hasattr(gene_expr, 'toarray'):
        gene_expr = gene_expr.toarray().flatten()
    else:
        gene_expr = gene_expr.flatten()
    
    # Create DataFrame for plotting individual cells
    plot_data = pd.DataFrame({
        'expression': gene_expr,
        'condition': data.obs['sample_type_rename'].values
    })
    
    # Remove zero values for cleaner visualization
    plot_data = plot_data[plot_data['expression'] > 0]
    
    def simple_swarm_with_overlap(y_values, x_center, width=0.35):
        """Create swarm-like positions that allow overlap but show distribution shape"""
        if len(y_values) == 0:
            return np.array([])
        
        # Create bins based on y values to simulate swarm density
        y_min, y_max = y_values.min(), y_values.max()
        n_bins = min(50, len(y_values) // 20 + 5)  # Adaptive number of bins
        
        # Create histogram to get density at each y level
        hist, bin_edges = np.histogram(y_values, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        x_positions = np.zeros(len(y_values))
        
        for i, y in enumerate(y_values):
            # Find which bin this y value falls into
            bin_idx = np.digitize(y, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, len(hist) - 1)
            
            # Get density at this y level
            density = hist[bin_idx]
            
            # Create width based on density (more points = wider spread)
            local_width = width * min(1.0, np.sqrt(density) / 10)
            
            # Random position within the local width (allows overlap)
            x_offset = np.random.uniform(-local_width/2, local_width/2)
            x_positions[i] = x_center + x_offset
        
        return x_positions
    
    # Plot individual cells with custom swarm allowing overlap
    for i, condition in enumerate(['Ctrl', 'EuE']):
        condition_data = plot_data[plot_data['condition'] == condition]['expression'].values
        
        # Create swarm positions that show distribution shape but allow overlap
        x_coords = simple_swarm_with_overlap(condition_data, i, width=0.35)
        
        # Plot individual cells as small gray dots
        ax.scatter(x_coords, condition_data,
                  color='lightgray', s=6, alpha=0.8,
                  edgecolors='none', zorder=1)
    
    # Add box plots for quartiles (clean style)
    if not pseudobulk_df.empty:
        ctrl_values = pseudobulk_df[pseudobulk_df['condition'] == 'Ctrl']['mean_expression'].values
        eue_values = pseudobulk_df[pseudobulk_df['condition'] == 'EuE']['mean_expression'].values
        
        if len(ctrl_values) > 0 and len(eue_values) > 0:
            box_data = [ctrl_values, eue_values]
            box_parts = ax.boxplot(box_data, positions=[0, 1], widths=0.3, 
                                  patch_artist=False, showfliers=False,
                                  boxprops=dict(color='black', linewidth=2),
                                  whiskerprops=dict(color='black', linewidth=2),
                                  capprops=dict(color='black', linewidth=2),
                                  medianprops=dict(color='black', linewidth=2),
                                  zorder=5)
    
    # Overlay sample pseudobulk means as larger black dots
    for i, condition in enumerate(['Ctrl', 'EuE']):
        condition_data = pseudobulk_df[pseudobulk_df['condition'] == condition]
        if not condition_data.empty:
            # Add slight jitter to x-coordinates for sample dots
            jitter = np.random.normal(0, 0.08, len(condition_data))
            x_coords = np.full(len(condition_data), i) + jitter
            
            # Plot sample means as larger black dots
            ax.scatter(x_coords, condition_data['mean_expression'],
                      color='black', s=60, alpha=0.8,
                      edgecolors='black', linewidth=0.5, zorder=10)
    
    # Statistical test and annotation
    if not pseudobulk_df.empty:
        ctrl_values = pseudobulk_df[pseudobulk_df['condition'] == 'Ctrl']['mean_expression'].values
        eue_values = pseudobulk_df[pseudobulk_df['condition'] == 'EuE']['mean_expression'].values
        
        if len(ctrl_values) > 0 and len(eue_values) > 0:
            statistic, p_value = mannwhitneyu(ctrl_values, eue_values, alternative='two-sided')
            # Use the global y_max if provided, otherwise calculate locally
            sig_y_position = y_max * 1.05 if y_max is not None else max(plot_data['expression'].max(), pseudobulk_df['mean_expression'].max()) * 1.05
            add_significance_annotation(ax, 0, 1, sig_y_position, p_value)
    
    # Formatting to match original style with correct labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control', 'Endo'], fontsize=16)
    ax.set_ylabel(f'{gene} Expression (log(counts+1))', fontsize=18, fontweight='bold')
    ax.set_title(f'{gene}', fontsize=24, fontweight='bold', pad=30)
    
    # Set y-axis to start from 0 like RPA2, with consistent max if provided
    if y_max is not None:
        ax.set_ylim(bottom=0, top=y_max * 1.3)  # Add some padding for significance annotation
    else:
        ax.set_ylim(bottom=0)
    
    # Clean up the plot appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
    ax.tick_params(axis='y', which='major', labelsize=16)
    
    plt.tight_layout()
    
    # Use styler's finish_plot method for SVG-optimized saving
    styler.finish_plot(save_plots=True, save_dir=os.path.dirname(save_path_base), 
                      name=os.path.basename(save_path_base))
    
    return fig, ax

# Pre-calculate max y-value across all genes for consistent y-axis
print("Pre-calculating y-axis range across all genes...\n")
max_y_value = 0
gene_data_cache = {}

for gene in genes:
    if gene not in stromal_filtered.var_names:
        print(f"Gene {gene} not found in dataset.")
        continue
    
    # Filter to cells with non-zero expression
    gene_expression = stromal_filtered[:, gene].X
    if hasattr(gene_expression, 'toarray'):
        gene_expression = gene_expression.toarray().flatten()
    else:
        gene_expression = gene_expression.flatten()
    
    non_zero_mask = gene_expression > 0
    data_gene = stromal_filtered[non_zero_mask, :].copy()
    
    if data_gene.n_obs == 0:
        continue
    
    # Store data for later use
    gene_data_cache[gene] = data_gene
    
    # Calculate pseudobulk for this gene
    pseudobulk_df = pd.DataFrame()
    if sample_id_col:
        pseudobulk_df = calculate_pseudobulk_stats(data_gene, gene, sample_id_col)
    
    # Get max values for this gene
    gene_expr = data_gene[:, gene].X
    if hasattr(gene_expr, 'toarray'):
        gene_expr = gene_expr.toarray().flatten()
    else:
        gene_expr = gene_expr.flatten()
    
    # Individual cell max
    cell_max = gene_expr.max()
    
    # Pseudobulk max
    pseudobulk_max = 0
    if not pseudobulk_df.empty:
        pseudobulk_max = pseudobulk_df['mean_expression'].max()
    
    # Update global max
    gene_max = max(cell_max, pseudobulk_max)
    max_y_value = max(max_y_value, gene_max)
    print(f"  {gene}: max value = {gene_max:.3f}")

print(f"\nGlobal max y-value across all genes: {max_y_value:.3f}")
print("This will be used for consistent y-axis scaling.\n")

# Main analysis loop
all_results = []
all_pseudobulk_data = []

print("Creating publication-ready plots...\n")

for gene in genes:
    if gene not in stromal_filtered.var_names:
        print(f"Gene {gene} not found in dataset.")
        continue
    
    print(f"Processing {gene}...")
    
    # Use cached data if available
    if gene in gene_data_cache:
        data_gene = gene_data_cache[gene]
    else:
        # Filter to cells with non-zero expression
        gene_expression = stromal_filtered[:, gene].X
        if hasattr(gene_expression, 'toarray'):
            gene_expression = gene_expression.toarray().flatten()
        else:
            gene_expression = gene_expression.flatten()
        
        non_zero_mask = gene_expression > 0
        data_gene = stromal_filtered[non_zero_mask, :].copy()
    
    print(f"  {data_gene.n_obs} cells with non-zero expression")
    
    if data_gene.n_obs == 0:
        print(f"  No cells with expression for {gene}")
        continue
    
    # Calculate pseudobulk statistics
    pseudobulk_df = pd.DataFrame()
    if sample_id_col:
        pseudobulk_df = calculate_pseudobulk_stats(data_gene, gene, sample_id_col)
        all_pseudobulk_data.extend(pseudobulk_df.to_dict('records'))
        
        # Print summary statistics
        for condition in ['Ctrl', 'EuE']:
            cond_data = pseudobulk_df[pseudobulk_df['condition'] == condition]
            if not cond_data.empty:
                print(f"  {condition}: {len(cond_data)} samples, "
                      f"mean={cond_data['mean_expression'].mean():.3f} ± "
                      f"{cond_data['mean_expression'].sem():.3f}")
    
    # Create publication plots
    for font_family in ['Helvetica', 'Arial']:
        suffix = f"_{font_family.lower()}"
        
        if not pseudobulk_df.empty:
            plot_path = os.path.join(plot_dir, f'{gene}_publication{suffix}')
            fig, ax = create_publication_plot(data_gene, gene, pseudobulk_df, plot_path, font_family, max_y_value)

    # Statistical analysis (run once per gene, not per font/color combination)
    if not pseudobulk_df.empty:
        ctrl_values = pseudobulk_df[pseudobulk_df['condition'] == 'Ctrl']['mean_expression']
        eue_values = pseudobulk_df[pseudobulk_df['condition'] == 'EuE']['mean_expression']
        
        if len(ctrl_values) > 0 and len(eue_values) > 0:
            statistic, p_value = mannwhitneyu(ctrl_values, eue_values, alternative='two-sided')
            
            # Effect size
            pooled_std = np.sqrt((np.var(ctrl_values) + np.var(eue_values)) / 2)
            cohens_d = np.nan
            if pooled_std > 0:
                cohens_d = (np.mean(eue_values) - np.mean(ctrl_values)) / pooled_std
            
            result = {
                'gene': gene,
                'ctrl_n_samples': len(ctrl_values),
                'eue_n_samples': len(eue_values),
                'ctrl_mean': np.mean(ctrl_values),
                'ctrl_sem': ctrl_values.sem() if hasattr(ctrl_values, 'sem') else np.std(ctrl_values)/np.sqrt(len(ctrl_values)),
                'eue_mean': np.mean(eue_values),
                'eue_sem': eue_values.sem() if hasattr(eue_values, 'sem') else np.std(eue_values)/np.sqrt(len(eue_values)),
                'mann_whitney_statistic': statistic,
                'p_value': p_value,
                'cohens_d': cohens_d
            }
            all_results.append(result)
            
            print(f"  Statistical test: p={p_value:.4f}, Cohen's d={cohens_d:.3f}")
    
    print(f"  Plots saved for {gene}\n")

# Save results
if all_results:
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(plot_dir, 'publication_statistical_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"Statistical results saved to: {results_csv_path}")

if all_pseudobulk_data:
    pseudobulk_df_all = pd.DataFrame(all_pseudobulk_data)
    pseudobulk_csv_path = os.path.join(plot_dir, 'publication_pseudobulk_data.csv')
    pseudobulk_df_all.to_csv(pseudobulk_csv_path, index=False)
    print(f"Pseudobulk data saved to: {pseudobulk_csv_path}")

print("\n" + "="*50)
print("PUBLICATION PLOTS COMPLETED")
print("="*50)
print("Generated plots:")
print("1. Single-cell violin plots showing cell-level distributions")
print("2. Pseudobulk violin plots showing sample-level statistics")
print("3. Statistical results CSV")
print("4. Pseudobulk sample data CSV")
print("5. All plots saved as both PNG and SVG for Illustrator editing")
print("\nReady for interactive exploration...")

# Start interactive session for further iteration
# embed()