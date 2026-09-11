# trainer/trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Train.gnn_model import ImprovedProteinGNN
from Train.metrics import find_best_threshold_by_mcc, calculate_metrics
from Train.seed import set_seed
from Train.logger import CSVLogger
from Train.config import Config
from Train.losses import DualViewLoss, InfoNCELoss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        set_seed(self.config.seed)
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.base_path = self.config.base_path
        os.makedirs(self.base_path, exist_ok=True)

    def train_fold(self, fold, train_graph, val_graph, target_etype, epochs=None, batch_size=None):
        set_seed(self.config.seed)
        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size

        model = ImprovedProteinGNN().to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='max',  # AUC越大越好
                                      factor=0.5,  # 每次减半
                                      patience=5,  # 5轮不提升就降学习率
                                      min_lr=1e-6)  # 最小学习率

        loss_fn = DualViewLoss(
            eta=self.config.loss_eta,
            auc_margin=self.config.loss_auc_margin
        )
        aux_weight = self.config.aux_weight
        aux_edge_sample = self.config.aux_edge_sample
        aux_neg_ratio = self.config.aux_neg_ratio
        contrastive_weight = self.config.contrastive_weight
        contrastive_drop_rate = self.config.contrastive_drop_rate
        contrastive_temperature = self.config.contrastive_temperature
        aux_loss_fn = nn.BCEWithLogitsLoss()
        contrastive_loss_fn = InfoNCELoss(temperature=contrastive_temperature)
        aux_internal_etypes = [
            # ('peptide', 'pp_interacts', 'peptide'),
            # ('receptor', 'rr_interacts', 'receptor'),
            ('peptide', 'e_interact', 'peptide'),
            ('receptor', 'e_interact', 'receptor')
        ]

        def _sample_positive_edges(graph, etype):
            num_edges = graph.number_of_edges(etype)
            if num_edges == 0:
                return None, None
            if aux_edge_sample and num_edges > aux_edge_sample:
                idx = torch.randperm(num_edges, device=self.device)[:aux_edge_sample]
            else:
                idx = torch.arange(num_edges, device=self.device)
            src, dst = graph.edges(etype=etype)
            return src[idx], dst[idx]

        def _sample_negative_edges(graph, etype, num_samples):
            if num_samples <= 0:
                return None, None
            src_type, _, dst_type = etype
            num_src = graph.num_nodes(src_type)
            num_dst = graph.num_nodes(dst_type)
            src_list = []
            dst_list = []
            attempts = 0
            while len(src_list) < num_samples and attempts < 10:
                attempts += 1
                remain = num_samples - len(src_list)
                cand_src = torch.randint(0, num_src, (remain * 2,), device=self.device)
                cand_dst = torch.randint(0, num_dst, (remain * 2,), device=self.device)
                mask = ~graph.has_edges_between(cand_src, cand_dst, etype=etype)
                cand_src = cand_src[mask]
                cand_dst = cand_dst[mask]
                take = min(remain, cand_src.shape[0])
                if take > 0:
                    src_list.append(cand_src[:take])
                    dst_list.append(cand_dst[:take])
            if not src_list:
                return None, None
            return torch.cat(src_list, dim=0), torch.cat(dst_list, dim=0)

        def _augment_graph_drop_edges(graph, drop_rate):
            if drop_rate <= 0:
                return graph
            masks = {}
            for etype in graph.canonical_etypes:
                num_edges = graph.number_of_edges(etype)
                if num_edges == 0:
                    masks[etype] = torch.zeros(0, dtype=torch.bool, device=self.device)
                    continue
                keep_mask = torch.rand(num_edges, device=self.device) >= drop_rate
                if keep_mask.sum() == 0:
                    keep_mask[torch.randint(0, num_edges, (1,), device=self.device)] = True
                masks[etype] = keep_mask
            return dgl.edge_subgraph(graph, masks, relabel_nodes=False)

        def _masked_edge_pred_loss(graph):
            losses = []
            for etype in aux_internal_etypes:
                if etype not in graph.canonical_etypes:
                    continue
                pos_src, pos_dst = _sample_positive_edges(graph, etype)
                if pos_src is None:
                    continue
                num_pos = pos_src.shape[0]
                num_neg = int(num_pos * aux_neg_ratio)
                neg_src, neg_dst = _sample_negative_edges(graph, etype, num_neg)
                if neg_src is None:
                    continue
                h2 = model.node_embeddings(graph, feat_dropout=0.0, exclude_etype=etype)
                pos_logits = model.edge_logits_from_pairs(
                    h2, etype[0], etype[2], pos_src, pos_dst, head="aux"
                )
                neg_logits = model.edge_logits_from_pairs(
                    h2, etype[0], etype[2], neg_src, neg_dst, head="aux"
                )
                logits = torch.cat([pos_logits, neg_logits], dim=0)
                labels = torch.cat([
                    torch.ones_like(pos_logits),
                    torch.zeros_like(neg_logits)
                ], dim=0)
                losses.append(aux_loss_fn(logits, labels))
            if not losses:
                return torch.tensor(0.0, device=self.device)
            return torch.stack(losses).mean()

        def _contrastive_loss(graph):
            if contrastive_weight <= 0:
                return torch.tensor(0.0, device=self.device)
            g_aug = _augment_graph_drop_edges(graph, contrastive_drop_rate)
            h_main = model.node_embeddings(graph, feat_dropout=0.0)
            h_aug = model.node_embeddings(g_aug, feat_dropout=0.0)
            losses = []
            for ntype in graph.ntypes:
                if h_main[ntype].numel() == 0:
                    continue
                z1 = model.project_embeddings(h_main[ntype])
                z2 = model.project_embeddings(h_aug[ntype])
                losses.append(contrastive_loss_fn(z1, z2))
            if not losses:
                return torch.tensor(0.0, device=self.device)
            return torch.stack(losses).mean()

        train_graph = train_graph.to(self.device)
        val_graph = val_graph.to(self.device)

        logger = CSVLogger(self.base_path, prefix=f"fold_{fold}")
        logger.write_headers()

        best_val_auc = 0.0
        early_stop_counter = 0
        patience = self.config.patience

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_bce_loss = 0.0
            total_auc_loss = 0.0
            total_aux_loss = 0.0
            total_edges = train_graph.number_of_edges(target_etype)
            if total_edges == 0:
                raise ValueError("No training edges for target etype.")
            indices = torch.randperm(total_edges).to(self.device)

            for i in range(0, total_edges, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_logits = model(train_graph, target_etype)[batch_indices]
                batch_labels = train_graph.edges[target_etype].data['label'][batch_indices].float().to(self.device)

                # loss = loss_fn(batch_logits, batch_labels)
                main_loss, bce_loss, auc_loss = loss_fn(batch_logits, batch_labels)
                aux_edge_loss_value = torch.tensor(0.0, device=self.device)
                contrastive_loss_value = torch.tensor(0.0, device=self.device)
                if aux_weight > 0:
                    aux_edge_loss_value = _masked_edge_pred_loss(train_graph)
                if contrastive_weight > 0:
                    contrastive_loss_value = _contrastive_loss(train_graph)

                loss = main_loss + aux_weight * aux_edge_loss_value + contrastive_weight * contrastive_loss_value

                total_loss += loss.item() * len(batch_indices)
                total_bce_loss += bce_loss.item() * len(batch_indices)
                total_auc_loss += auc_loss.item() * len(batch_indices)
                total_aux_loss += (aux_weight * aux_edge_loss_value.item() + contrastive_weight * contrastive_loss_value.item()) * len(batch_indices)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / total_edges if total_edges > 0 else 0.0
            avg_bce_loss = total_bce_loss / total_edges if total_edges > 0 else 0.0
            avg_auc_loss = total_auc_loss / total_edges if total_edges > 0 else 0.0
            avg_aux_loss = total_aux_loss / total_edges if total_edges > 0 else 0.0

            # evaluation
            model.eval()
            with torch.no_grad():
                train_logits = model(train_graph, target_etype)
                train_probs = torch.sigmoid(train_logits).cpu().numpy()
                train_labels = train_graph.edges[target_etype].data['label'].cpu().numpy()

                val_logits = model(val_graph, target_etype)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_labels = val_graph.edges[target_etype].data['label'].cpu().numpy()

                threshold = find_best_threshold_by_mcc(val_labels, val_probs)
                val_metrics = calculate_metrics(val_labels, val_probs, threshold)

                val_logits_tensor = val_logits.clone().detach()
                val_labels_tensor = torch.tensor(val_labels).float().to(self.device)
                val_loss, val_bce, val_auc = loss_fn(val_logits_tensor, val_labels_tensor)
                val_metrics['loss'] = val_loss.item()
                val_metrics['bce_loss'] = val_bce.item()
                val_metrics['auc_loss'] = val_auc.item()

                val_aux_loss = 0.0
                if aux_weight > 0 or contrastive_weight > 0:
                    edge_aux = _masked_edge_pred_loss(val_graph).item() if aux_weight > 0 else 0.0
                    contrastive_aux = _contrastive_loss(val_graph).item() if contrastive_weight > 0 else 0.0
                    val_aux_loss = aux_weight * edge_aux + contrastive_weight * contrastive_aux
                val_metrics['aux_loss'] = val_aux_loss

                train_metrics = calculate_metrics(train_labels, train_probs, threshold)

            print(
                f"Fold {fold} Epoch {epoch:03d} | Loss: {avg_loss:.4f} (BCE: {avg_bce_loss:.4f}, AUC: {avg_auc_loss:.4f}, AUX: {avg_aux_loss:.4f}) | "
                f"Train AUC: {train_metrics['auc']:.4f} | Val AUC: {val_metrics['auc']:.4f}")

            # print(
            #     f"Fold {fold} Epoch {epoch:03d} | Loss: {avg_loss:.4f} "
            #     f"Train AUC: {train_metrics['auc']:.4f} | Val AUC: {val_metrics['auc']:.4f}")

            # write logs
            train_row = ",".join(map(str, [
                epoch, avg_loss,
                train_metrics['precision'], train_metrics['recall'], train_metrics['f1'],
                train_metrics['accuracy'], train_metrics['auc'], train_metrics['aupr'], train_metrics['mcc'],
                train_metrics['TP'], train_metrics['TN'], train_metrics['FP'], train_metrics['FN'],
                optimizer.param_groups[0]['lr'], avg_aux_loss
            ]))
            val_row = ",".join(map(str, [
                epoch, val_metrics['loss'],
                val_metrics['precision'], val_metrics['recall'], val_metrics['f1'],
                val_metrics['accuracy'], val_metrics['auc'], val_metrics['aupr'], val_metrics['mcc'],
                val_metrics['TP'], val_metrics['TN'], val_metrics['FP'], val_metrics['FN'],
                optimizer.param_groups[0]['lr'], val_metrics['aux_loss']
            ]))
            logger.append_train(train_row)
            logger.append_val(val_row)

            scheduler.step(val_metrics['auc'])

            # early stop logic
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                torch.save(model.state_dict(), os.path.join(self.base_path, f"fold{fold}_best_model.pth"))
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # scheduler step at end epoch if desired
            # scheduler.step()

        return best_val_auc