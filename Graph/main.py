from pathlib import Path

OUTPUT_DIR = "output/"

GRAPH_FILE_NAME = "graph.dgl"

PEP_COS="features/peptide_cos.pkl"
PRO_COS="features/protein_cos.pkl"

PEP_E_INTERACT = "edges/peptide_e_interact.csv"
PRO_E_INTERACT = "edges/protein_e_interact.csv"

T5_PEP = "features/peptide_t5.pkl"
T5_REC = "features/protein_t5.pkl"


print('准备构建图')
from GraphEdgeBuilder import EdgeBuilder
builder = EdgeBuilder(device=0)
builder.load_samples(
    positive_file=Path(OUTPUT_DIR) / "positive_pairs.txt",
    negative_file=Path(OUTPUT_DIR) / "negative_pairs.txt"
)

builder.construct_graph(
    pep_edges_file=PEP_E_INTERACT,
    pro_edges_file=PRO_E_INTERACT,
    peptide_cos_file=PEP_COS,
    protein_cos_file=PRO_COS,
    pep_threshold_low=0.915,
    pep_threshold_high=10,
    pro_threshold_low=0.82,
    pro_threshold_high=10,
)

from PretrainedFeatureManager import PretrainedFeatureManager
manager = PretrainedFeatureManager(
    G=builder.G,
    peptide_to_id=builder.peptide_to_id,
    receptor_to_id=builder.receptor_to_id,
    peptide_feature_file=T5_PEP,
    receptor_feature_file=T5_REC,
    save_dir=OUTPUT_DIR,
    graph_file_name=GRAPH_FILE_NAME
)
manager.set_pretrained_features()
manager.remove_nodes_without_features()
manager.save_graph()
