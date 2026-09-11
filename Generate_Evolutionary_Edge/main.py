import argparse
import os
import sys
import json

# 导入自定义处理模块
from sequence_preprocessor import SequencePreprocessor
from psi_blast_runner import PsiBlastRunner
from blast_result_extractor import BlastResultExtractor
from blast_result_parser import BlastResultParser
from evolution_edge_builder import EvolutionEdgeBuilder

parser = argparse.ArgumentParser(description="Generate embeddings and compute similarity")
parser.add_argument(
    "--type",
    choices=["protein", "peptide"],
    default="protein",
    help="Which sequence type to process (default: protein).",
)

args = parser.parse_args()


def main():
    """
    执行完整的同源性分析流程。
    """
    # 配置：定义输入/输出路径和外部依赖
    INPUT_CSV = f"{args.type}_sequences.csv"  # 输入：单列 CSV 格式的序列文件
    WORK_DIR = f"workdir_{args.type}"  # 所有中间/最终输出的根工作目录

    # 中间目录
    FASTA_DIR = os.path.join(WORK_DIR, "fasta")  # 每条序列的 FASTA 文件
    RESULT_DIR = os.path.join(WORK_DIR, "psiblast_results")  # PSI-BLAST 原始输出文件 (.txt)
    PSSM_DIR = os.path.join(WORK_DIR, "pssm")  # 位置特异性得分矩阵
    EXTRACTED_DIR = os.path.join(WORK_DIR, "extracted_rounds")  # 最后一轮 BLAST 结果
    EVALUE_DIR = os.path.join(WORK_DIR, "evalue_files")  # 每条查询序列的 E-value 列表（供边构建器使用）

    # 最终输出
    OUTPUT_JSON = os.path.join(WORK_DIR, "blast_hits.json")  # 解析后的命中结果 JSON 格式：{query_seq: {hit_seq: e_value}}
    EDGE_OUTPUT = os.path.join(WORK_DIR, "evolution_edges.csv")  # 供下游建模使用的最终边列表

    # 外部依赖：预格式化的 BLAST 蛋白质数据库路径（例如 nr, swissprot）
    # 注意：应指向由 `makeblastdb` 创建的数据库的基础名称（不含扩展名）
    BLAST_DB_PATH = f"{args.type}/database/{args.type}"  # ←←← 必须更新为实际的数据库路径！

    ERROR_LOG = os.path.join(WORK_DIR, "psiblast_errors.log")  # 记录 PSI-BLAST 运行失败的文件

    # 确保根工作目录存在
    os.makedirs(WORK_DIR, exist_ok=True)

    # 步骤 1：序列预处理
    #   - 从 CSV 读取序列
    #   - 去重并生成每条序列的 FASTA 文件
    #   - 构建映射关系：sequence_id（例如 "sequence_1"）→ 氨基酸字符串
    print("=== 步骤 1：序列预处理 ===")
    if not os.path.exists(INPUT_CSV):
        print(f"[错误] 未找到输入 CSV 文件：{INPUT_CSV}")
        sys.exit(1)

    preprocessor = SequencePreprocessor(csv_path=INPUT_CSV, output_dir=FASTA_DIR)
    preprocessor.run()

    # 复用去重后的序列列表构建 ID 到序列的映射
    unique_sequences = preprocessor.load_sequences()
    fasta_mapping = {f"sequence_{i + 1}": seq for i, seq in enumerate(unique_sequences)}

    # 步骤 2：运行 PSI-BLAST
    #   - 对每个 FASTA 文件执行 PSI-BLAST（10 次迭代）
    #   - 保存比对结果和 PSSM 文件
    #   - 将失败的任务记录到 error_log
    print("\n=== 步骤 2：运行 PSI-BLAST ===")
    psiblast_runner = PsiBlastRunner(
        fasta_folder=FASTA_DIR,
        result_folder=RESULT_DIR,
        pssm_folder=PSSM_DIR,
        db_path=BLAST_DB_PATH,
        error_log=ERROR_LOG
    )
    psiblast_runner.run()

    # 步骤 3：提取最后一轮迭代结果
    #   - 从每个 *_results.txt 文件中提取从 "Results from round X" 开始的内容
    #     X 为执行的最后一轮迭代次数
    print("\n=== 步骤 3：提取最后一轮 PSI-BLAST 结果 ===")
    extractor = BlastResultExtractor(
        input_folder=RESULT_DIR,
        output_folder=EXTRACTED_DIR
    )
    extractor.run()

    # 步骤 4：解析显著命中结果并生成 JSON
    #   - 解析提取的文件，得到 {query_id: {hit_id: e_value}} 格式
    #   - 使用 fasta_mapping 将 ID 替换为实际的氨基酸序列
    #   - 输出为结构化 JSON
    print("\n=== 步骤 4：解析 BLAST 命中结果 ===")
    parser = BlastResultParser(
        extracted_dir=EXTRACTED_DIR,
        fasta_mapping=fasta_mapping,
        output_json=OUTPUT_JSON
    )
    parser.run()

    # 步骤 5：构建进化边（TSV 格式）
    #   - EvolutionEdgeBuilder 期望每个查询序列有一个 .txt 文件，格式为：
    #         hit_sequence_id, e_value
    #   - 由于解析器输出的是合并后的 JSON，我们首先将其转换为所需的按文件格式
    #   - 然后应用分数转换（默认：-log10(E)）并写入边
    print("\n=== 步骤 5：构建进化边 ===")
    print("  -> 将 JSON 转换为按查询序列的 E-value 文件...")

    os.makedirs(EVALUE_DIR, exist_ok=True)

    # 加载解析后的命中结果（现在使用实际序列作为键）
    with open(OUTPUT_JSON, "r") as f:
        all_hits = json.load(f)

    # 反向映射：序列字符串 → ID（用于重建原始 ID 以便文件命名）
    seq_to_id = {seq: seq_id for seq_id, seq in fasta_mapping.items()}

    # 为 EdgeBuilder 生成每个查询序列对应的 .txt 文件
    for query_seq, hits in all_hits.items():
        query_id = seq_to_id.get(query_seq)
        if query_id is None:
            continue  # 如果查询序列不在原始映射中则跳过（理论上不应发生）
            #出现了Query，Score这样的，但是会直接跳过

        out_path = os.path.join(EVALUE_DIR, f"{query_id}.txt")
        with open(out_path, "w") as f_out:
            for hit_seq, e_val in hits.items():
                hit_id = seq_to_id.get(hit_seq, hit_seq)  # 如果 ID 缺失则回退使用序列本身
                f_out.write(f"{hit_id}, {e_val}\n")

    # 现在运行边构建器
    edge_builder = EvolutionEdgeBuilder(
        input_dir=EVALUE_DIR,
        seq_mapping=fasta_mapping,
        output_file=EDGE_OUTPUT,
        transform='neg_log10'  # 推荐：分数越高表示越显著
    )
    edge_builder.run()

    # 完成
    print("\n 所有步骤成功完成！")
    print(f"最终进化边保存至：{EDGE_OUTPUT}")


if __name__ == "__main__":
    main()