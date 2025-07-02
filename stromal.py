import scanpy as sc
import pandas as pd
import os
import anndata as ad
import numpy as np
from IPython import embed

# Load the Stromal cells AnnData object
stromal_path = '/Volumes/hunter_ssd/endometrial_scRNA/analysis/tn_2022/endo-2022_stromal.h5ad'
stromal = ad.read_h5ad(stromal_path)
plot_dir = '/Volumes/hunter_ssd/endometrial_scRNA/analysis/tn_2022/plots/'
os.makedirs(plot_dir, exist_ok=True)

# look at the values for each: 'celltype', 'celltype_main', 'subtypes', 'sample_type_rename'
print("Unique values in 'celltype':", stromal.obs['celltype'].unique())
print("Unique values in 'celltype_main':", stromal.obs['celltype_main'].unique())
print("Unique values in 'subtypes':", stromal.obs['subtypes'].unique())
print("Unique values in 'sample_type_rename':", stromal.obs['sample_type_rename'].unique())

# Filter data to only include Ctrl and EuE samples
ctrl_eue_mask = stromal.obs['sample_type_rename'].isin(['Ctrl', 'EuE'])
stromal_filtered = stromal[ctrl_eue_mask, :].copy()

print(f"Filtered data: {stromal_filtered.n_obs} cells (Ctrl and EuE only) out of {stromal.n_obs} total cells")

# Check if we have a sample/patient identifier
print("Available obs columns:", stromal_filtered.obs.columns.tolist())
print("Checking for sample identifiers...")

# Look for sample/patient ID column (common names)
sample_id_col = None
for col in ['sample_id', 'patient_id', 'PID', 'sample', 'patient', 'donor_id']:
    if col in stromal_filtered.obs.columns:
        sample_id_col = col
        print(f"Found sample ID column: {col}")
        break

if sample_id_col:
    print(f"Unique samples in {sample_id_col}:", stromal_filtered.obs[sample_id_col].unique())
    print(f"Sample counts by condition:")
    print(stromal_filtered.obs.groupby(['sample_type_rename', sample_id_col]).size().unstack(fill_value=0))

# Pseudobulk analysis and violin plots
import scipy.stats as stats
from scipy.stats import mannwhitneyu

# Initialize lists to store results for CSV output
all_results = []
all_pseudobulk_data = []

for gene in ['NBN', 'RAD50', 'MRE11']:
    if gene in stromal_filtered.var_names:
        print(f"\n=== Analysis for {gene} ===")
        
        # Create a copy of the data for this specific gene
        data_gene = stromal_filtered.copy()
        
        # Filter out cells with zero expression for this specific gene
        gene_expression = data_gene[:, gene].X
        if hasattr(gene_expression, 'toarray'):
            gene_expression = gene_expression.toarray().flatten()
        else:
            gene_expression = gene_expression.flatten()
        
        # Keep only cells with non-zero expression for this gene
        non_zero_mask = gene_expression > 0
        data_gene = data_gene[non_zero_mask, :]
        
        print(f"Gene {gene}: {non_zero_mask.sum()} cells with non-zero expression out of {len(non_zero_mask)} total cells")
        
        if data_gene.n_obs > 0:
            # Create violin plot
            sc.pl.violin(data_gene, keys=gene, groupby='sample_type_rename', 
                        stripplot=True, jitter=True, inner='box', 
                        show=False, save=f'_{gene}_violin_stromal_ctrl_vs_eue.png')
            default_path = os.path.join(os.getcwd(), 'figures', f'violin_{gene}_violin_stromal_ctrl_vs_eue.png')
            target_path = os.path.join(plot_dir, f'violin_{gene}_violin_stromal_ctrl_vs_eue.png')
            if os.path.exists(default_path):
                os.rename(default_path, target_path)
            
            # Pseudobulk analysis if we have sample IDs
            if sample_id_col:
                # Create pseudobulk data by averaging expression per sample
                pseudobulk_data = []
                
                for condition in ['Ctrl', 'EuE']:
                    condition_data = data_gene[data_gene.obs['sample_type_rename'] == condition]
                    for sample in condition_data.obs[sample_id_col].unique():
                        sample_data = condition_data[condition_data.obs[sample_id_col] == sample]
                        if sample_data.n_obs > 0:
                            # Get expression for this gene in this sample
                            sample_expr = sample_data[:, gene].X
                            if hasattr(sample_expr, 'toarray'):
                                sample_expr = sample_expr.toarray().flatten()
                            else:
                                sample_expr = sample_expr.flatten()
                            
                            # Calculate mean expression for this sample
                            mean_expr = np.mean(sample_expr)
                            sample_pseudobulk = {
                                'gene': gene,
                                'sample_id': sample,
                                'condition': condition,
                                'mean_expression': mean_expr,
                                'n_cells': sample_data.n_obs
                            }
                            pseudobulk_data.append(sample_pseudobulk)
                            all_pseudobulk_data.append(sample_pseudobulk)
                
                if pseudobulk_data:
                    pseudobulk_df = pd.DataFrame(pseudobulk_data)
                    print(f"Pseudobulk data for {gene}:")
                    print(pseudobulk_df)
                    
                    # Statistical test on pseudobulk data
                    ctrl_values = pseudobulk_df[pseudobulk_df['condition'] == 'Ctrl']['mean_expression'].values
                    eue_values = pseudobulk_df[pseudobulk_df['condition'] == 'EuE']['mean_expression'].values
                    
                    if len(ctrl_values) > 0 and len(eue_values) > 0:
                        # Mann-Whitney U test (non-parametric)
                        statistic, p_value = mannwhitneyu(ctrl_values, eue_values, alternative='two-sided')
                        
                        print(f"Pseudobulk statistical test for {gene}:")
                        print(f"  Ctrl samples (n={len(ctrl_values)}): mean={np.mean(ctrl_values):.3f}, std={np.std(ctrl_values):.3f}")
                        print(f"  EuE samples (n={len(eue_values)}): mean={np.mean(eue_values):.3f}, std={np.std(eue_values):.3f}")
                        print(f"  Mann-Whitney U statistic: {statistic}")
                        print(f"  p-value: {p_value:.6f}")
                        
                        # Effect size (Cohen's d equivalent for Mann-Whitney)
                        pooled_std = np.sqrt((np.var(ctrl_values) + np.var(eue_values)) / 2)
                        cohens_d = np.nan
                        if pooled_std > 0:
                            cohens_d = (np.mean(eue_values) - np.mean(ctrl_values)) / pooled_std
                            print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
                        
                        # Store results for CSV
                        result = {
                            'gene': gene,
                            'ctrl_n_samples': len(ctrl_values),
                            'eue_n_samples': len(eue_values),
                            'ctrl_mean': np.mean(ctrl_values),
                            'ctrl_std': np.std(ctrl_values),
                            'eue_mean': np.mean(eue_values),
                            'eue_std': np.std(eue_values),
                            'mann_whitney_statistic': statistic,
                            'p_value': p_value,
                            'cohens_d': cohens_d,
                            'total_cells_analyzed': data_gene.n_obs,
                            'cells_with_nonzero_expression': non_zero_mask.sum()
                        }
                        all_results.append(result)
            else:
                print("No sample ID column found - cannot perform pseudobulk analysis")
                print("Consider adding sample/patient identifiers to perform proper statistical testing")
        else:
            print(f"No cells with non-zero expression found for gene {gene}")
    else:
        print(f"Gene {gene} not found in dataset.")

# Save results to CSV files
if all_results:
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(plot_dir, 'pseudobulk_statistical_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nStatistical results saved to: {results_csv_path}")

if all_pseudobulk_data:
    pseudobulk_df_all = pd.DataFrame(all_pseudobulk_data)
    pseudobulk_csv_path = os.path.join(plot_dir, 'pseudobulk_sample_data.csv')
    pseudobulk_df_all.to_csv(pseudobulk_csv_path, index=False)
    print(f"Pseudobulk sample data saved to: {pseudobulk_csv_path}")

print("\nViolin plots and pseudobulk analysis completed")

embed()

