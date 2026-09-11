# 简介

本仓库包含了 **SEER-PEPPI** 框架的 PyTorch 实现，该框架基于序列和深度学习预测肽-蛋白质相互作用。

## 目录

- [安装指南](#安装指南)
- [训练流程](#训练流程)
- [推理流程](#推理流程)

## 安装指南

### 1. 创建基础环境

本项目使用 conda 管理环境，基础依赖已写在 `environment.yml` 中：

```bash
conda env create -f environment.yml
conda activate SEER-PEPPI 
```

### 2. 安装 DGL 2.4.0

#### Windows 用户

由于 DGL 官方自 2024.06.27 起不再提供 Windows 和 macOS 的预编译包，Windows 用户需要从源码编译安装。

在 SEER-PEPPI 环境下运行以下命令

```bash
# 1. 下载 DGL 2.4.0 源码
curl -L https://codeload.github.com/dmlc/dgl/zip/refs/tags/v2.4.0 -o dgl-2.4.0.zip
tar -xf dgl-2.4.0.zip
cd dgl-2.4.0

# 2. 创建 build 目录并配置 CMake
mkdir build
cd build
cmake -DCMAKE_CXX_FLAGS="/DDGL_EXPORTS" -DCMAKE_CONFIGURATION_TYPES="Release" -DDMLC_FORCE_SHARED_CRT=ON -DUSE_CUDA=ON .. -G "Visual Studio 16 2019"

# 3. 编译
msbuild dgl.sln /m

# 4. 安装 Python 包
cd ..\python
python setup.py install
```

#### Linux 用户

```bash
conda install -c dglteam/label/th24_cu118 dgl
```

更详细的步骤可参照 [DGL 官方文档](https://dgl.org.cn/dgl_docs/install/index.html)。

### 3. 安装 BLAST+

#### Windows 用户

请从 NCBI 官网下载 Windows 版 BLAST+。

下载地址：  
https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.17.0+-win64.exe

#### Linux 用户

```bash
conda install -c bioconda blast=2.17.0
```

### 4. 安装 ProtT5 模型

从 https://zenodo.org/records/4644188 下载 prot_t5_xl_uniref50 模型，并将其解压到 `Generate_Similarity_Edge/prot_t5_xl_uniref50` 文件夹中。

## 项目结构
```
SEER-PEPPI/
├── Data/                          # 数据集
├── Generate_Evolutionary_Edge/    # 基于 PSI-BLAST 构建进化边
├── Generate_Similarity_Edge/      # 基于 ProT5 构建语义相似度边
├── Graph/                         # 异构图构建
├── Train/                         # 模型训练
├── Test/                          # 模型测试与评估
├── Results/                       # 训练输出结果
└── README.md                      # 项目说明
```

## 训练流程

#### 1. 数据预处理

```bash
cd Data
python click_me.py --fasta ${dataset}.fasta --output-dir ${output}
```

其中`${dataset}` 可以是 `Train_8622`、`Camp`、`Test_1440` 或 `Test_251`。一般保持 `${output}` 和 `${dataset}` 一致。

#### 2. 生成ProtT5嵌入并计算余弦相似度

将步骤1得到的 `Data/${output}/*.pkl` 文件(2个)放入 `Generate_Similarity_Edge` 文件夹中。运行以下命令：

```bash
cd Generate_Similarity_Edge
python generate.py --type ${type}
```

其中`${type}` 可以是 protein 或 peptide。需要分别为两类节点生成各自的嵌入和计算余弦相似度

#### 3. 计算进化相似性

将步骤1得到的 `Data/${output}/*.csv` 文件(2个)放入 `Generate_Evolutionary_Edge` 文件夹中，运行以下命令：  

```bash
cd Generate_Evolutionary_Edge
python main.py --type ${type}
```

其中`${type}` 可以是 protein 或 peptide。需要分别为两类节点生成进化相似性

若没有构建数据集，需要先构建数据库。进入`Generate_Evolutionary_Edge/workdir_${type}/fasta` 文件夹中，获取 `all_sequences.fasta` ，并移动到 `Generate_Evolutionary_Edge/${type}/database` 文件夹中，运行以下命令：

```bash
cd Generate_Evolutionary_Edge/${type}/database
makeblastdb -dbtype prot -in all_sequences.fasta -input_type fasta -parse_seqids -out ${type}
```

其中 `${type}` 可以是 protein 或 peptide。同样的，需要分别为两类节点生成各自的数据库。构建完毕后再次运行 `main.py` 文件

#### 4. 构建异构图

将步骤1得到的 `Data/${output}/*.txt` 文件(2个)放入 `Graph/output` 文件夹中。将步骤2得到的 `Generate_Similarity_Edge/*.pkl` 文件(4个)放入 `Graph/features` 文件夹中。将步骤3得到的 `Generate_Evolutionary_Edge/workdir_${type}/evolution_edges.csv` 文件(2个)放入 `Graph/edges` 文件夹中。

运行以下命令：

```bash
cd Graph
python main.py
```

#### 5. 模型训练

在 `Train` 文件夹内运行以下命令：

```bash
cd Train
python main.py
```

结果保留在 `Results` 文件夹中。

## 推理流程

推理流程与训练流程类似。遵循训练流程的步骤1-4，得到异构图后运行以下命令：

```bash
cd Test
python main.py
```

您可以使用自己的数据集从头训练或推理，务必确保数据集格式与提供的示例一致，路径设置正确。不同数据集构建图的阈值不同，您可以参照我们的论文提供的最佳阈值，也可自由测试。

## 联系
我们欢迎您在使用过程中提出问题或建议。您可以通过以下方式联系我们：shushengzhao@stu.ouc.edu.cn