def build_adj_list(edge_index, num_nodes):
    """
    edge_index: [2, num_edges]
    返回：邻接表 adj，每个节点对应邻居列表
    """
    adj = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        adj[src].append(dst)
    return adj