# ecole polytechnique - c.durr - 2009

# Kuhn-Munkres, The hungarian algorithm.  Complexity O(n^3)
# Computes a max weight perfect matching in a bipartite graph
# for min weight matching, simply negate the weights.

""" Global variables:
       n = number of vertices on each side
       U,V vertex sets
       lu,lv are the labels of U and V resp.
       the matching is encoded as 
       - a mapping Mu from U to V, 
       - and Mv from V to U.

    The algorithm repeatedly builds an alternating tree, rooted in a
    free vertex u0. S is the set of vertices in U covered by the tree.
    For every vertex v, T[v] is the parent in the tree and Mv[v] the
    child.
    The algorithm maintains minSlack, s.t. for every vertex v not in
    T, minSlack[v]=(val,u1), where val is the minimum slack
    lu[u]+lv[v]-w[u][v] over u in S, and u1 is the vertex that
    realizes this minimum.
    Complexity is O(n^3), because there are n iterations in
    maxWeightMatching, and each call to augment costs O(n^2). This is
    because augment() makes at most n iterations itself, and each
    updating of minSlack costs O(n).
    """


def improveLabels(val):
    """ change the labels, and maintain minSlack. 
    """
    for u in S:
        lu[u] -= val
    for v in V:
        if v in T:
            lv[v] += val
        else:
            minSlack[v][0] -= val


def improveMatching(v):
    """ apply the alternating path from v to the root in the tree. 
    """
    u = T[v]
    if u in Mu:
        improveMatching(Mu[u])
    Mu[u] = v
    Mv[v] = u


def slack(u, v): return lu[u] + lv[v] - w[u][v]


def augment():
    """ augment the matching, possibly improving the lablels on the way.
    """
    while True:
        # select edge (u,v) with u in S, v not in T and min slack
        ((val, u), v) = min([(minSlack[v], v) for v in V if v not in T])
        assert u in S
        if val > 0:
            improveLabels(val)
        # now we are sure that (u,v) is saturated
        assert slack(u, v) == 0
        T[v] = u  # add (u,v) to the tree
        if v in Mv:
            u1 = Mv[v]  # matched edge,
            assert not u1 in S
            S[u1] = True  # ... add endpoint to tree
            for v in V:  # maintain minSlack
                if not v in T and minSlack[v][0] > slack(u1, v):
                    minSlack[v] = [slack(u1, v), u1]
        else:
            improveMatching(v)  # v is a free vertex
            return


def maxWeightMatching(weights):
    """ given w, the weight matrix of a complete bipartite graph,
        returns the mappings Mu : U->V ,Mv : V->U encoding the matching
        as well as the value of it.
    """
    global U, V, S, T, Mu, Mv, lu, lv, minSlack, w
    w = weights
    n = len(w)
    U = V = range(n)
    lu = [max([w[u][v] for v in V]) for u in U]  # start with trivial labels
    lv = [0 for v in V]
    Mu = {}  # start with empty matching
    Mv = {}
    while len(Mu) < n:
        free = [u for u in V if u not in Mu]  # choose free vertex u0
        u0 = free[0]
        S = {u0: True}  # grow tree from u0 on
        T = {}
        minSlack = [[slack(u0, v), u0] for v in V]
        augment()
    # val. of matching is total edge weight
    val = sum(lu) + sum(lv)
    return (Mu, Mv, val)



    #  a small example

    # print maxWeightMatching([[1,2,3,4],[2,4,6,8],[3,6,9,12],[4,8,12,16]])