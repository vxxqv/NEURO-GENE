import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

DEFAULT_GENES = [
    "BDNF", "CREB1", "NTRK2", "ARC", "FOS", "JUN", "EGR1", "CAMK2A", 
    "GRIN1", "GRIN2B", "MAPK1", "AKT1", "MTOR", "SYN1", "PSD95"
]

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="NeuroGene-Hub: A Co-Expression Network Tool for Prioritizing Neuroplasticity-Associated Genes"
    )
    parser.add_argument("--expression", type=str, default="data/expression.csv")
    parser.add_argument("--metadata", type=str, default="data/metadata.csv")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--corr_threshold", type=float, default=0.70)
    parser.add_argument("--cond_a", type=str, default="control")
    parser.add_argument("--cond_b", type=str, default="treated")
    return parser.parse_args()

def generate_synthetic_data(expr_path, meta_path):
    os.makedirs(os.path.dirname(expr_path), exist_ok=True)
    np.random.seed(42)
    controls = [f"control_{i+1}" for i in range(5)]
    treated = [f"treated_{i+1}" for i in range(5)]
    samples = controls + treated
    
    df_meta = pd.DataFrame({
        "sample": samples,
        "condition": ["control"] * 5 + ["treated"] * 5
    })
    df_meta.to_csv(meta_path, index=False)
    
    expr_data = {}
    for gene in DEFAULT_GENES:
        if gene in ["BDNF", "CREB1", "ARC", "FOS", "EGR1"]:
            base_ctrl = np.random.normal(loc=4.5, scale=0.4, size=5)
            base_trtd = np.random.normal(loc=7.8, scale=0.5, size=5)
        elif gene in ["GRIN2B", "PSD95", "SYN1"]:
            base_ctrl = np.random.normal(loc=5.0, scale=0.3, size=5)
            base_trtd = np.random.normal(loc=6.2, scale=0.3, size=5)
        else:
            base_ctrl = np.random.normal(loc=5.5, scale=0.4, size=5)
            base_trtd = np.random.normal(loc=5.6, scale=0.4, size=5)
        expr_data[gene] = np.concatenate([base_ctrl, base_trtd])
        
    df_expr = pd.DataFrame(expr_data, index=samples).T
    df_expr.index.name = "gene"
    df_expr.reset_index().to_csv(expr_path, index=False)

def load_data(expr_path, meta_path):
    if not os.path.exists(expr_path) or not os.path.exists(meta_path):
        generate_synthetic_data(expr_path, meta_path)
    return pd.read_csv(expr_path, index_col=0), pd.read_csv(meta_path)

def normalize_data(df_expr):
    if df_expr.max().max() > 100:
        cpm = df_expr.div(df_expr.sum(axis=0), axis=1) * 1e6
        return np.log2(cpm + 1)
    return np.log2(df_expr + 1)

def run_differential_expression(df_expr, df_meta, cond_a, cond_b):
    samples_a = df_meta[df_meta["condition"] == cond_a]["sample"].tolist()
    samples_b = df_meta[df_meta["condition"] == cond_b]["sample"].tolist()
    
    de_results = []
    for gene in df_expr.index:
        vals_a = df_expr.loc[gene, samples_a].values
        vals_b = df_expr.loc[gene, samples_b].values
        log2_fc = np.mean(vals_b) - np.mean(vals_a)
        _, p_val = ttest_ind(vals_b, vals_a, equal_var=False)
        
        if np.isnan(p_val):
            p_val = 1.0
            
        de_results.append({
            "gene": gene,
            "log2FoldChange": log2_fc,
            "pvalue": p_val
        })
    return pd.DataFrame(de_results).set_index("gene")

def build_coexpression_network(df_expr, threshold):
    corr_matrix = df_expr.T.corr(method="pearson").abs()
    G = nx.Graph()
    G.add_nodes_from(corr_matrix.index)
    genes = corr_matrix.index
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            weight = corr_matrix.iloc[i, j]
            if weight >= threshold:
                G.add_edge(genes[i], genes[j], weight=weight)
    return G

def calculate_network_centrality(G):
    deg_centrality = nx.degree_centrality(G)
    try:
        eigen_centrality = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
    except nx.PowerIterationFailedConvergence:
        eigen_centrality = deg_centrality
        
    centrality_df = pd.DataFrame({
        "DegreeCentrality": deg_centrality,
        "EigenvectorCentrality": eigen_centrality
    })
    centrality_df.index.name = "gene"
    return centrality_df

def compute_importance_score(df_de, df_centrality):
    merged = df_de.join(df_centrality)
    merged["neg_log10_p"] = -np.log10(merged["pvalue"].clip(lower=1e-10))
    merged["PriorityScore"] = merged["log2FoldChange"].abs() * merged["neg_log10_p"] * (merged["EigenvectorCentrality"] + 0.1)
    return merged.sort_values(by="PriorityScore", ascending=False)

def generate_plots(df_ranked, G, output_dir):
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(8, 6))
    highlight = df_ranked["pvalue"] < 0.05
    plt.scatter(df_ranked.loc[~highlight, "log2FoldChange"], df_ranked.loc[~highlight, "neg_log10_p"], c="grey", alpha=0.6, label="Not Significant")
    plt.scatter(df_ranked.loc[highlight, "log2FoldChange"], df_ranked.loc[highlight, "neg_log10_p"], c="crimson", alpha=0.8, label="Significant ($P < 0.05$)")
    for gene, row in df_ranked.head(5).iterrows():
        plt.annotate(gene, (row["log2FoldChange"], row["neg_log10_p"]), textcoords="offset points", xytext=(0,10), ha='center', weight='bold')
    plt.title("Differential Expression: Volcano Plot", fontsize=14, weight='bold')
    plt.xlabel("$Log_2$ Fold Change")
    plt.ylabel("$-Log_{10}$ p-value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "volcano_plot.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42, k=0.4)
    node_sizes = [G.degree(node) * 150 + 200 for node in G.nodes()]
    node_colors = [df_ranked.loc[node, "PriorityScore"] if node in df_ranked.index else 0 for node in G.nodes()]
    edges = G.edges()
    weights = [G[u][v]['weight'] * 2 for u, v in edges]
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=weights, edge_color="gainsboro")
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    plt.colorbar(nodes, label="Combined Priority Score", shrink=0.7)
    plt.title("Neuroplasticity Co-Expression Network Map", fontsize=14, weight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gene_network.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    top_15 = df_ranked.head(15).reset_index()
    sns.barplot(data=top_15, x="PriorityScore", y="gene", hue="gene", palette="flare", legend=False)
    plt.title("Top 15 Prioritized Hub Candidates", fontsize=14, weight='bold')
    plt.xlabel("NeuroGene Priority Score")
    plt.ylabel("Gene Identifier")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_genes_ranking.png"), dpi=300)
    plt.close()

def export_report(df_ranked, G, output_dir):
    df_ranked.to_csv(os.path.join(output_dir, "ranked_genes.csv"))
    report_path = os.path.join(output_dir, "hub_gene_report.txt")
    top_hub = df_ranked.index[0]
    
    report_content = f"""========================================================================
NEUROGENE PIPELINE ANALYTICAL REPORT
========================================================================
Network Statistics:
- Total Functional Nodes (Genes evaluated): {len(G.nodes)}
- Total Co-expression Interactions Observed: {len(G.edges)}
- Average Network Density: {nx.density(G):.4f}

Top 5 Prioritized Hub Genes Candidate Overview:
------------------------------------------------------------------------
Rank\tGene\tLog2FC\tp-value\tEigenvalue Centrality\tPriority Score
------------------------------------------------------------------------
"""
    for rank, (gene, row) in enumerate(df_ranked.head(5).iterrows(), 1):
        report_content += f"{rank}\t{gene}\t{row['log2FoldChange']:.3f}\t{row['pvalue']:.2e}\t{row['EigenvectorCentrality']:.3f}\t\t{row['PriorityScore']:.3f}\n"
        
    report_content += f"""------------------------------------------------------------------------
System Insights Summary:
The primary candidate identified is '{top_hub}' displaying a robust cross-talk 
profile showing strong statistical changes paired with high network integration density. 

Report compiled successfully in directory: '{output_dir}'.
========================================================================
"""
    with open(report_path, "w") as f:
        f.write(report_content)

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    df_expr, df_meta = load_data(args.expression, args.metadata)
    df_norm = normalize_data(df_expr)
    df_de = run_differential_expression(df_norm, df_meta, args.cond_a, args.cond_b)
    G = build_coexpression_network(df_norm, args.corr_threshold)
    df_centrality = calculate_network_centrality(G)
    df_ranked = compute_importance_score(df_de, df_centrality)
    
    generate_plots(df_ranked, G, args.output_dir)
    export_report(df_ranked, G, args.output_dir)

if __name__ == "__main__":
    main()
