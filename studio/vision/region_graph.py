"""
Grafo de Adjacência de Regiões (RAG) e Fusão Adaptativa com Base em CIEDE2000.
Implementa a Seção 13 do Plano de Evolução Completo (Region Adjacency Graph).
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
from ..core.models import RegionFeatures
from .perceptual_color import PerceptualColorEngine

class RegionAdjacencyGraph:
    def __init__(self, features: List[RegionFeatures], labels_map: np.ndarray):
        self.features = {f.id: f for f in features}
        self.labels_map = labels_map
        self.adjacency: Dict[int, set] = {f.id: set() for f in features}
        self._build_adjacency()

    def _build_adjacency(self):
        """Encontra vizinhos de borda através de deslocamentos espaciais na matriz de labels."""
        lm = self.labels_map
        # Pares de labels distintos entre pixels vizinhos horizontais e verticais (vetorizado)
        diff_h = lm[:, :-1] != lm[:, 1:]
        diff_v = lm[:-1, :] != lm[1:, :]
        u_all = np.concatenate([lm[:, :-1][diff_h], lm[:-1, :][diff_v]]).astype(np.int64)
        v_all = np.concatenate([lm[:, 1:][diff_h], lm[1:, :][diff_v]]).astype(np.int64)
        if u_all.size:
            pairs = np.unique(np.stack([np.minimum(u_all, v_all), np.maximum(u_all, v_all)], axis=1), axis=0)
            for u, v in pairs:
                u, v = int(u), int(v)
                if u in self.adjacency and v in self.adjacency:
                    self.adjacency[u].add(v)
                    self.adjacency[v].add(u)

        for f_id, neighbors in self.adjacency.items():
            if f_id in self.features:
                self.features[f_id].neighbors = list(neighbors)

    def adaptive_merge(self, max_delta_e: float = 4.5) -> Tuple[np.ndarray, List[RegionFeatures]]:
        """
        Funde iterativamente superpixels vizinhos que possuem distância de cor perceptual
        menor que o limiar estrito max_delta_e, evitando descaracterizar bordas fortes.
        """
        merged_labels = self.labels_map.copy()
        uf_parent = {f_id: f_id for f_id in self.features}

        def find(u):
            while uf_parent[u] != u:
                uf_parent[u] = uf_parent[uf_parent[u]]
                u = uf_parent[u]
            return u

        def union(u, v):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                # Mantém como raiz a região com maior área
                if self.features[root_u].area >= self.features[root_v].area:
                    uf_parent[root_v] = root_u
                else:
                    uf_parent[root_u] = root_v

        # Avalia arestas
        for u in self.features:
            for v in self.adjacency[u]:
                if u >= v:
                    continue
                root_u = find(u)
                root_v = find(v)
                if root_u == root_v:
                    continue
                de = PerceptualColorEngine.delta_e_ciede2000(
                    self.features[root_u].mean_lab,
                    self.features[root_v].mean_lab
                )
                if de < max_delta_e:
                    union(root_u, root_v)

        # Mapeia novos labels
        unique_roots = {}
        new_id = 0
        final_features = []

        h, w = merged_labels.shape
        # Atualiza a matriz de labels
        for u in self.features:
            root = find(u)
            if root not in unique_roots:
                unique_roots[root] = new_id
                new_id += 1

        # Remapeia matriz via tabela de consulta (labels sem feature vão para 0, como antes)
        lut = np.zeros(int(self.labels_map.max()) + 1, dtype=np.int32)
        for u in self.features:
            lut[u] = unique_roots[find(u)]
        remapped = lut[self.labels_map]

        # Estatísticas por região em uma passada (área, centroide e bbox)
        flat = remapped.ravel()
        ys, xs = np.indices((h, w))
        counts = np.bincount(flat, minlength=new_id)
        sum_y = np.bincount(flat, weights=ys.ravel(), minlength=new_id)
        sum_x = np.bincount(flat, weights=xs.ravel(), minlength=new_id)
        from scipy import ndimage
        slices = ndimage.find_objects(remapped + 1)

        # Agrupa os membros de cada região fundida para recálculo ponderado (Seção 11.4 da Auditoria Real)
        members_by_root: Dict[int, List[RegionFeatures]] = {}
        for u, f in self.features.items():
            r = find(u)
            members_by_root.setdefault(r, []).append(f)

        # Recalcula características para cada região fundida com média ponderada pela área real
        for root, assigned_id in unique_roots.items():
            area = int(counts[assigned_id])
            if area == 0:
                continue
            cy = int(sum_y[assigned_id] / area)
            cx = int(sum_x[assigned_id] / area)
            sl_y, sl_x = slices[assigned_id]
            bbox = (sl_x.start, sl_y.start, sl_x.stop - sl_x.start, sl_y.stop - sl_y.start)

            members = members_by_root.get(root, [self.features[root]])
            tot_sub_area = sum(m.area for m in members)
            if tot_sub_area > 0:
                w_l = sum(m.mean_lab[0] * m.area for m in members) / tot_sub_area
                w_a = sum(m.mean_lab[1] * m.area for m in members) / tot_sub_area
                w_b = sum(m.mean_lab[2] * m.area for m in members) / tot_sub_area
                w_lab = (float(w_l), float(w_a), float(w_b))
                w_sal = float(sum(m.saliency * m.area for m in members) / tot_sub_area)
                w_angle = float(sum(m.texture_angle * m.area for m in members) / tot_sub_area)
                w_edge = float(sum(m.edge_density * m.area for m in members) / tot_sub_area)
                w_var = float(sum(m.lab_variance * m.area for m in members) / tot_sub_area)
            else:
                w_lab = self.features[root].mean_lab
                w_sal = self.features[root].saliency
                w_angle = self.features[root].texture_angle
                w_edge = self.features[root].edge_density
                w_var = self.features[root].lab_variance

            rf = RegionFeatures(
                region_id=assigned_id,
                id=assigned_id,
                mean_lab=w_lab,
                lab_variance=w_var,
                texture_angle=w_angle,
                edge_density=w_edge,
                area=area,
                centroid=(cx, cy),
                bbox=bbox,
                neighbors=[],
                saliency=w_sal
            )
            final_features.append(rf)

        return remapped, final_features
