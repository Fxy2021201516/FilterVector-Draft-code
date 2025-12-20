# SmartRoute: Filtered Approximate Nearest Neighbor Search

This project implements **SmartRoute**, a Filtered Approximate Nearest Neighbor Search (LANNS) framework optimized for scenarios with **Diverse Labels**.

SmartRoute integrates multiple search strategies and utilizes a lightweight machine learning model as a **Router**. It dynamically selects the optimal search path based on the filtering conditions of a query in real-time. This project also includes complete implementations of **UNG (Unified Navigating Graph)** and **ACORN** as baselines or components.

## 1\. Core Algorithm Introduction

### 1.1 SmartRoute (Proposed)

SmartRoute addresses the performance bottlenecks of single-index solutions under varying filter selectivity and label complexity through an intelligent routing mechanism.

  * **Intelligent Router**: SmartRoute trains a lightweight **Random Decision Forest** model. Based on query features extracted in real-time, it dynamically selects one of the following three strategies:

    1.  **UNG-nTfalse (UNG-style)**: Performs search on the **Unified Graph**. Suitable for scenarios where the proportion of filtered vectors is small and the number of Entry Label Sets (ELS) is manageable.
    2.  **ACORN-gamma (ACORN-style)**: Performs search on a predicate-agnostic proximity graph, ignoring nodes that do not satisfy the label constraints during the search. Suitable for scenarios where the proportion of filtered vectors is large.
    3.  **ACORN-gamma-improved (Hybrid)**: Performs search on the predicate-agnostic proximity graph but evaluates neighbors even if they do not satisfy the label constraints (to avoid recall degradation caused by graph disconnection).

  * **Key Technical Optimizations**:

      * **IntelELS**: A fast Entry Label Set (ELS) retrieval algorithm combining a Trie tree and a prediction model. It intelligently chooses between Top-down or Bottom-up traversal strategies, significantly reducing query preprocessing latency.
      * **Fast Feature Extraction**: Utilizes **Roaring Bitmaps** to accelerate set operations, allowing for rapid estimation of key routing features like the proportion of vectors passing the filter.

### 1.2 UNG (Unified Navigating Graph)

  * **Mechanism**: Adopts a "Pre-filtering" strategy. It constructs a Label Navigating Graph (LNG) to encode label containment relationships and combines intra-group proximity graphs with Cross-group Edges to build a unified graph index.
  * **Characteristics**: Performs excellently in scenarios with fewer label types or simple logic, avoiding visits to invalid vectors.

### 1.3 ACORN

  * **Mechanism**: Adopts a "Predicate-Agnostic" strategy. By increasing the connection density (Densification) of the HNSW graph, it ensures subgraph connectivity under arbitrary label filters.
  * **Characteristics**: Simple index construction and high search efficiency when filter selectivity is high (i.e., many vectors pass the filter).

## 2\. Project Structure

The codebase is designed modularly to support hybrid compilation and independent execution:

```text
SmartRoute
├── ACORN/              # Core implementation of the ACORN algorithm
├── UNG/                # Core implementation of the UNG algorithm
├── RF-selector/        # Training and inference scripts for the Router (Random Forest)
├── scripts/            # Automated experiment scripts
│   ├── exp.sh             # [Core Entry Point] One-click build, GT generation, and search
│   ├── build_hybrid.sh    # Hybrid index construction script
│   ├── generate_gt.sh     # Ground Truth generation tool
│   ├── generate_queries.sh# Independent query generation and analysis tool
│   └── search.sh          # Search performance evaluation script
├── Dockerfile          # Containerized environment definition
└── requirements.txt    # Python dependencies
```

## 3\. Environment Setup (Docker)

To ensure consistency in the experimental environment, using Docker is recommended.

### 3.1 Build Image

```bash
docker build -t SmartRoute .
```

### 3.2 Run Container

When starting the container, mount the data directory and the result output directory. Assuming the host data is at `/my/local/data` and results should be stored in `/my/local/results`:

```bash
docker run -it --rm \
  -v /my/local/data:/data \
  -v /my/local/results:/results \
  SmartRoute /bin/bash
```
## 4. Dataset Preparation

### 4.1 Download Data
We provide the necessary datasets (including base vectors, vector labels, query vectors, and query labels for 8 datasets) hosted on Hugging Face.

* **Hugging Face Repository**: [paper-review/8-Datasets](https://huggingface.co/datasets/paper-review/8-Datasets)


### 4.2 Directory Structure
After downloading, please organize your data according to the following hierarchy to ensure the scripts can locate them correctly (using `Dataset_Name` as a generic example.):

```text
/data/Dataset_Name/
├── Dataset_Name_base.fvecs         # Base vectors (float32)
├── Dataset_Name_base_labels.txt    # Label IDs corresponding to base vectors
├── query_task_001/                 # Directory for specific query tasks
│   ├── Dataset_Name_query.fvecs
│   └── Dataset_Name_query_labels.txt
└── ...
```


## 5\. Core Experiment Execution (Exp.sh)

Use `scripts/exp.sh` with a JSON configuration file to complete the full experiment pipeline (Index Construction -\> GT Generation -\> Search Evaluation).

### 5.1 Run Command

```bash
# Ensure scripts have execution permissions
chmod +x scripts/*.sh

# Run the main experiment
./scripts/exp.sh experiments.json
```

### 5.2 Configuration Example (`experiments.json`)

```json
{
   "experiments": [
      {
         "dataset_name": "Your_Dataset_Name",
         "shared_config": {
            "data_dir": "/data/Your_Dataset_Name",
            "output_dir": "/results/",
            "build_mode": "serial",
            "max_degree": 32,
            "Lbuild": 100,
            "alpha": 1.2,
            "num_cross_edges": 6,
            "num_entry_points": 16,
            "K": 10,
            "acorn_params": {
               "N": 1000000,
               "M": 32,
               "M_beta": 64,
               "gamma": 80
            }
         },
         "tasks": [
            {
               "query_dir_name": "query_task_001",
               "acorn_search_params": {
                  "acorn_efs_start": 100,
                  "acorn_efs_step_slow": 100,
                  "acorn_efs_step_fast": 100
               },
               "algorithms": [ "UNG-nTfalse", "ACORN-gamma" ]
            }
         ]
      }
   ]
}
```

-----

## 6\. Independent Query Generation & Analysis

In addition to using existing query sets, this project provides the `scripts/generate_queries.sh` tool for generating synthetic queries, subset queries, or analyzing the properties of existing queries (e.g., selectivity distribution).

### 6.1 Run Command

The script accepts an independent JSON configuration file as input:

```bash
# Usage: ./scripts/generate_queries.sh [config_file] [build_dir]
# Note: build_dir is the output directory for compilation artifacts, usually /app/build_gene (inside container)

./scripts/generate_queries.sh generate_queries_config.json /app/build_gene
```

### 6.2 Supported Modes

The tool supports the following `mode` options:

  * **`weighted_sub_base`**: Weighted sampling, supporting control over query length and label distribution.
  * **`analyze_only`**: Analyzes statistical characteristics of existing query files.

### 6.3 Configuration Example (`generate_queries_config.json`)

The following configuration demonstrates how to generate weighted subset queries of different lengths and analyze them. **Note: Paths must be adjusted according to Docker mount points.**

```json
{
   "query_tasks": [
      {
         "enabled": true,
         "task_name": "analyze_existing_queries",
         "mode": "analyze_only",
         "dataset": "Your_Dataset_Name",
         "data_dir": "/data/Your_Dataset_Name",
         "overwrite": true,
         "analysis_params": {
            "analyze": true,
            "candidate_file": "/data/Your_Dataset_Name/query_task_001/Your_Dataset_Name_query_labels.txt",
            "profiled_output": "/data/Your_Dataset_Name/query_task_001/profiled_stats.csv"
         }
      },
      {
         "enabled": true,
         "task_name": "weighted_sub_base_len_2",
         "mode": "weighted_sub_base",
         "dataset": "Your_Dataset_Name",
         "data_dir": "/data/Your_Dataset_Name",
         "overwrite": true,
         "generation_params": {
            "num_points": 500,
            "K": 10,
            "truncate_to_fixed_length": true,
            "num_labels_per_query": 2,
            "expected_num_label": 2
         },
         "sub_base_params": {
            "num_points": 500,
            "query_length": 2,
            "K": 10,
            "min_children": 0,
            "max_coverage": 10000000,
            "cache-file": "/data/Your_Dataset_Name/sub_base_cache"
         },
         "analysis_params": {
            "analyze": true
         }
      }
   ]
}
```

-----

## 7\. Result Output Structure

Experiment results will be saved in the configured output directory with the following hierarchy:

```text
/results/Your_Dataset_Name/
├── Index/              # Serialized index files (UNG & ACORN)
├── GroundTruth/        # Generated Ground Truth binary files
└── Results/            # Search performance logs and metrics
    ├── UNG-nTfalse/
    ├── ACORN-gamma/
    ├── ACORN-gamma-improved/
    └── SmartRoute/
        └── Index_GT_Search_Params/  # Folder containing detailed parameters
            ├── results/             # Performance statistics (CSV/Logs)
            └── others/              # Execution logs (search_output.txt)
```