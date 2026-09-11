# 蛋白质-肽相互作用预测的主评估脚本
import os
import argparse
import dgl
from Test.seed import SeedSetter
from Test.test import ModelEvaluator
from Test.summarizer import FiveFoldSummarizer
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def main():
    parser = argparse.ArgumentParser(description="在验证图上评估预训练的GNN模型。")
    parser.add_argument("--graph", type=str, default=r"D:\pyprogram\fanhua\graph\output\graph\graph_1440.dgl",
                        help="DGL图文件路径（例如：./data/graph.dgl）")
    parser.add_argument("--model_dir", type=str, default=r"D:\pyprogram\edge\result_V4",
                        help="包含5个模型文件的目录：fold1_best_model.pth ... fold5_best_model.pth")
    parser.add_argument("--output_dir", type=str, default=r"C:\Users\omen\Desktop\泛化实验结果",
                        help="保存评估结果的目录（默认：./results)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="运行设备（例如：'cuda:0' 或 'cpu'）")
    parser.add_argument("--seed", type=int, default=913,
                        help="随机种子，用于结果可复现（默认：913）")

    args = parser.parse_args()

    # 步骤1：设置随机种子
    SeedSetter.set_seed(args.seed)

    # 步骤2：准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 步骤3：加载图（在此仅评估模式下，假设训练和验证使用相同的图）
    print(f"正在从 {args.graph} 加载图...")
    graph = dgl.load_graphs(args.graph)[0][0]


    # # 步骤5：定义模型路径（期望 fold0 到 fold4）
    model_paths = [
        os.path.join(args.model_dir, f"fold{i}_best_model.pth") for i in range(0, 5)
    ]
    for mp in model_paths:
        if not os.path.exists(mp):
            raise FileNotFoundError(f"未找到模型文件：{mp}")

    # 步骤6：使用FoldTrainer运行评估
    print("正在评估模型...")
    target_etype = ('receptor', 'binds', 'peptide')
    trainer = ModelEvaluator(base_path=args.output_dir, device=args.device)
    trainer.test(
        graph=graph,
        target_etype=target_etype,
        model_paths=model_paths
    )

    # 步骤7：汇总结果
    print("正在生成汇总报告...")
    FiveFoldSummarizer.summarize(args.output_dir)

    print("评估成功完成！")


if __name__ == "__main__":
    main()