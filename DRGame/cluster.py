import time
import dgl
import numpy as np
import torch
import torch.nn as nn
from dgi import DGI
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from dgl import backend as F
import pickle
from sklearn.cluster import KMeans
import pandas as pd

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.asarray(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return np.asarray(features.todense())

def items_embeds(graph_path, feat_path, embeddings_path, args, offset):
    with open(graph_path, 'r') as f:
        lines = f.readlines()
    edges = []
    for line in lines:
        src, dst = map(int, line.strip().split(','))
        edges.append((src, dst + offset))
    g = dgl.graph(edges)
    g = dgl.to_bidirected(g)
    # g = dgl.remove_self_loop(g)
    # g = dgl.add_self_loop(g)
    print('Number of nodes:', g.number_of_nodes())
    print('Number of edges:', g.number_of_edges())
    print('Edges:', g.edges())    
    data = np.genfromtxt(feat_path, delimiter=',', dtype=int)
    rows, cols = data[:, 0], data[:, 1]
    mat = csr_matrix((np.ones_like(rows), (rows, cols)))    
    mat = mat.astype(np.float64)
    g.ndata['feat'] = F.tensor(preprocess_features(mat), dtype=F.data_type_dict['float32'])
    features = torch.FloatTensor(g.ndata["feat"])
    in_feats = features.shape[1]
    n_edges = g.num_edges()

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
    if args.gpu >= 0:
        g = g.to(args.gpu)
    # create DGI model
    dgi = DGI(
        g,
        in_feats,
        args.n_hidden,
        args.n_layers,
        nn.PReLU(args.n_hidden),
        args.dgi_dropout,
    )    
    if cuda:
        dgi.cuda()

    dgi_optimizer = torch.optim.Adam(
        dgi.parameters(), lr=args.dgi_lr, weight_decay=args.dgi_weight_decay)

    # train deep graph infomax
    cnt_wait = 0
    best = 1e9
    best_t = 0
    dur = []
    for epoch in range(args.n_dgi_epochs):
        dgi.train()
        if epoch >= 3:
            t0 = time.time()

        dgi_optimizer.zero_grad()
        loss = dgi(features)
        loss.backward()
        dgi_optimizer.step()

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(dgi.state_dict(), "best_dgi.pkl")
        else:
            cnt_wait += 1

        if cnt_wait == args.dgi_patience:
            print("Early stopping!")
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        print(
            "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
            "ETputs(KTEPS) {:.2f}".format(
                epoch, np.mean(dur), loss.item(), n_edges / np.mean(dur) / 1000
            )
        )
    print("Loading {}th epoch".format(best_t))
    dgi.load_state_dict(torch.load("best_dgi.pkl"))
    embeds = dgi.encoder(features, corrupt=False)
    embeds = embeds.detach()

    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeds, f)

    return embeds


def k_cluster(embeddings, k):
    embeddings = embeddings.cpu()

    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(embeddings)
    labels = kmeans.labels_

    node_ids = [str(i) for i in range(len(embeddings))]

    label_encoder = pd.factorize(labels)
    node_subject = label_encoder[0]

    return node_ids, node_subject


def read_items_embeddings(embeddings_path):
    with open(embeddings_path, 'rb') as f:
        items_embeddings = pickle.load(f)
    return items_embeddings