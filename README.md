# NEURO-GENE

**NEURO-GENE** is a lightweight Python bioinformatics tool for prioritizing neuroplasticity-associated genes. It combines differential expression analysis with Pearson co-expression networks and NetworkX centrality scoring to identify potential hub genes that may play stronger roles in neuroplasticity-related pathways.

The project is built around a simple pipeline: load or generate expression data, normalize it, compare control vs treated samples, build a co-expression network, calculate gene centrality, and rank genes using a combined priority score.

## Features

* Differential expression analysis between two sample conditions
* Pearson correlation-based co-expression network construction
* Degree and eigenvector centrality scoring using NetworkX
* Combined hub-gene priority ranking
* Automatic synthetic dataset generation if no input data is provided
* Clean output plots and text reports

## Outputs

Running the pipeline generates:

```text
outputs/
├── ranked_genes.csv
├── hub_gene_report.txt
├── volcano_plot.png
├── gene_network.png
└── top_genes_ranking.png
```

These outputs help visualize which genes show stronger expression changes, stronger network connectivity, and higher overall priority as candidate regulators.

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/vxxqv/NEURO-GENE.git
cd NEURO-GENE
pip install numpy pandas scipy networkx matplotlib seaborn
```

## Usage

Run the default pipeline:

```bash
python neurogene.py
```

By default, the script looks for:

```text
data/expression.csv
data/metadata.csv
```

If these files are not found, NEURO-GENE automatically creates a small synthetic dataset and runs the full analysis.

You can also provide custom settings:

```bash
python neurogene.py --expression data/expression.csv --metadata data/metadata.csv --output_dir outputs --corr_threshold 0.70 --cond_a control --cond_b treated
```

## Project Structure

```text
NEURO-GENE/
├── neurogene.py
├── outputs/
│   ├── ranked_genes.csv
│   ├── hub_gene_report.txt
│   ├── volcano_plot.png
│   ├── gene_network.png
│   └── top_genes_ranking.png
└── LICENSE
```

## Why This Project Matters

Neuroplasticity is controlled by many interacting genes rather than one isolated marker. NEURO-GENE gives a compact way to rank genes not only by expression change, but also by their importance inside a co-expression network. This makes the project useful as a simple computational biology framework for exploring candidate hub genes in neuroscience datasets.

## License

This project is licensed under the MIT License.
