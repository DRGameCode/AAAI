import torch.nn as nn
import torch as th


class RecLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

    def reduction(self, nodes):
        mail = nodes.mailbox['s']
        _, _, feature_size = mail.shape

        mask_mail = nodes.mailbox['msk'].unsqueeze(-1).repeat(1, 1, feature_size)
        mail = mail * mask_mail
        mail = mail.sum(dim = 1)
        return {'h': mail}

    def category_aggregation(self, edges):
        return {'s': edges.src['h'], 'msk': edges.data['mask']}


    def forward(self, graph, h, etype):
        with graph.local_scope():
            src, _ , dst = etype 
            feat_src = h[src]
            feat_dst = h[dst]

            
            degs = graph.out_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = th.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.nodes[src].data['h'] = feat_src
            
            graph.update_all(self.category_aggregation, self.reduction, etype = etype)

            
            rst = graph.nodes[dst].data['h']
            
            degs = graph.in_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm  
            return rst  
