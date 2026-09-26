# Introduction

This repository contains a PyTorch implementation of the **SEER-PEPPI** framework, which predicts peptide-protein interactions based on sequences and deep learning.

## Table of Contents

- [Installation Guide](#installation-guide)
- [Training Pipeline](#training-pipeline)
- [Inference Pipeline](#inference-pipeline)
- [Graph Construction Thresholds](#graph-construction-thresholds)

## Installation Guide

### 1. Create the Base Environment

This project uses conda to manage the environment. The base dependencies are listed in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate SEER-PEPPI 
```

### 2. Install DGL 2.4.0

#### Windows Users

Since DGL officially stopped providing precompiled packages for Windows and macOS as of 2024.06.27, Windows users need to compile and install from source.

Run the following commands in the SEER-PEPPI environment:

```bash
# 1. Download the DGL 2.4.0 source code
curl -L https://codeload.github.com/dmlc/dgl/zip/refs/tags/v2.4.0 -o dgl-2.4.0.zip
tar -xf dgl-2.4.0.zip
cd dgl-2.4.0

# 2. Create the build directory and configure CMake
mkdir build
cd build
cmake -DCMAKE_CXX_FLAGS="/DDGL_EXPORTS" -DCMAKE_CONFIGURATION_TYPES="Release" -DDMLC_FORCE_SHARED_CRT=ON -DUSE_CUDA=ON .. -G "Visual Studio 16 2019"

# 3. Compile
msbuild dgl.sln /m

# 4. Install the Python package
cd ..\python
python setup.py install
```

#### Linux Users

```bash
conda install -c dglteam/label/th24_cu118 dgl
```

For more detailed steps, refer to the [official DGL documentation](https://dgl.org.cn/dgl_docs/install/index.html).

### 3. Install BLAST+

#### Windows Users

Please download the Windows version of BLAST+ from the NCBI official website.

Download address:  
https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.17.0+-win64.exe

#### Linux Users

```bash
conda install -c bioconda blast=2.17.0
```

### 4. Install the ProtT5 Model

Download the `prot_t5_xl_uniref50` model from https://zenodo.org/records/4644188 and extract it into the `Generate_Similarity_Edge/prot_t5_xl_uniref50` folder.

## Project Structure

```
SEER-PEPPI/
├── Data/                          # Datasets
├── Generate_Evolutionary_Edge/    # Construct evolutionary edges based on PSI-BLAST
├── Generate_Similarity_Edge/      # Construct semantic similarity edges based on ProtT5
├── Graph/                         # Heterogeneous graph construction
├── Train/                         # Model training
├── Test/                          # Model testing and evaluation
├── Results/                       # Training output results
└── README.md                      # Project description
```

## Training Pipeline

#### 1. Data Preprocessing

```bash
cd Data
python click_me.py --fasta ${dataset}.fasta --output-dir ${output}
```

Here `${dataset}` can be `Train_8622`, `Camp`, `Test_1440`, or `Test_251`. In general, keep `${output}` the same as `${dataset}`.

#### 2. Generate ProtT5 Embeddings and Compute Cosine Similarity

Place the two `Data/${output}/*.pkl` files obtained in step 1 into the `Generate_Similarity_Edge` folder. Run the following command:

```bash
cd Generate_Similarity_Edge
python generate.py --type ${type}
```

Here `${type}` can be `protein` or `peptide`. You need to generate embeddings and compute cosine similarity separately for the two types of nodes.

#### 3. Compute Evolutionary Similarity

Place the two `Data/${output}/*.csv` files obtained in step 1 into the `Generate_Evolutionary_Edge` folder, and run the following command:

```bash
cd Generate_Evolutionary_Edge
python main.py --type ${type}
```

Here `${type}` can be `protein` or `peptide`. You need to generate evolutionary similarity separately for the two types of nodes.

If the dataset has not been constructed, you need to build the database first. Go to the `Generate_Evolutionary_Edge/workdir_${type}/fasta` folder, obtain `all_sequences.fasta`, and move it to the `Generate_Evolutionary_Edge/${type}/database` folder. Run the following command:

```bash
cd Generate_Evolutionary_Edge/${type}/database
makeblastdb -dbtype prot -in all_sequences.fasta -input_type fasta -parse_seqids -out ${type}
```

Here `${type}` can be `protein` or `peptide`. Similarly, you need to build separate databases for the two types of nodes. After construction is complete, run `main.py` again.

#### 4. Construct the Heterogeneous Graph

Place the two `Data/${output}/*.txt` files obtained in step 1 into the `Graph/output` folder. Place the four `Generate_Similarity_Edge/*.pkl` files obtained in step 2 into the `Graph/features` folder. Place the two `Generate_Evolutionary_Edge/workdir_${type}/evolution_edges.csv` files obtained in step 3 into the `Graph/edges` folder.

Run the following command:

```bash
cd Graph
python main.py
```

#### 5. Model Training

Run the following command in the `Train` folder:

```bash
cd Train
python main.py
```

Results are saved in the `Results` folder.

## Inference Pipeline

The inference pipeline is similar to the training pipeline. Follow steps 1-4 of the training pipeline. After obtaining the heterogeneous graph, run the following command:

```bash
cd Test
python main.py
```

You can use your own dataset to train from scratch or perform inference. Make sure the dataset format is consistent with the provided examples and the paths are set correctly. Different datasets have different thresholds for graph construction. You can use the optimal thresholds we provide or test them freely.

## Graph Construction Thresholds

The optimal thresholds for the Train_8622 dataset are:

| Edge Type                          | Number of Edges | Threshold |
|------------------------------------|-----------------|-----------|
| Peptide-peptide similarity edges   | 41513           | 0.9       |
| Protein-protein similarity edges   | 40104           | 0.99      |
| Peptide-peptide evolutionary edges | 42424           | 0.5       |
| Protein-protein evolutionary edges | 41504           | 1e-160    |

The optimal thresholds for the Camp dataset are:

| Edge Type                          | Number of Edges | Threshold |
|------------------------------------|-----------------|-----------|
| Peptide-peptide similarity edges   | 10067           | 0.915     |
| Protein-protein similarity edges   | 18365           | 0.82      |
| Peptide-peptide evolutionary edges | 10358           | 5e-4      |
| Protein-protein evolutionary edges | 18893           | 1e-65     |

The optimal thresholds for the Train_1440 dataset are:

| Edge Type                          | Number of Edges | Threshold |
|------------------------------------|-----------------|-----------|
| Peptide-peptide similarity edges   | 4884            | 0.865     |
| Protein-protein similarity edges   | 4824            | 0.985     |
| Peptide-peptide evolutionary edges | 4893            | 100       |
| Protein-protein evolutionary edges | 4775            | 1e-113    |

The optimal thresholds for the Train_251 dataset are:

| Edge Type                          | Number of Edges | Threshold |
|------------------------------------|-----------------|-----------|
| Peptide-peptide similarity edges   | 1382            | 0.82      |
| Protein-protein similarity edges   | 839             | 0.74      |
| Peptide-peptide evolutionary edges | 551             | 100       |
| Protein-protein evolutionary edges | 730             | 1e-50     |

## Contact

We welcome your questions or suggestions during use. You can contact us through:

shushengzhao@stu.ouc.edu.cn