import networkx as nx
import matplotlib.pyplot as plt
import random
import math
from write_graph import write_graphml_pos


def random_sum_list_old(n):
    """
    Generate a list of random integers that sum up to n.
    The length of the list is random, between 1 and one quarter of n, inclusive.
    The list will not contain any zeroes, and the minimum value is 2.
    :param n: the sum of the list
    :return: a list of random integers
    """
    max_length = n // 4  # maximum length is one quarter of n
    length = random.randint(1, max_length)
    if length == 1:
        return [n]
    else:
        total = 0
        result = []
        for i in range(length - 1):
            remaining = n - total - (length - i - 1) * 2  # minimum value is 2
            if remaining <= 0:
                x = 0  # add a zero if there is no remaining sum
            else:
                x = random.randint(2, remaining)
            result.append(x)
            total += x
        result.append(n - total)
        random.shuffle(result)
        return result
    
def random_sum_list(n):
    """
    Generate a list of random integers that sum up to n.
    The length of the list is random, between 3 and one quarter of n, inclusive.
    The list will not contain any zeroes, and the minimum value is 2.
    :param n: the sum of the list
    :return: a list of random integers
    """
    max_length = max(n // 4, 3)  # maximum length is one quarter of n or 3, whichever is greater
    length = random.randint(3, max_length)
    result = []
    while n > 0 and len(result) < length:
        if n - length + len(result) <= 0:
            x = n  # add the remaining sum if it's less than the minimum value
        else:
            max_val = min(n, (n - length + len(result)) * 2)
            if max_val < 3:
                x = n if len(result) == length - 1 else 3  # add the remaining sum or 3 if it's the last element
            else:
                x = random.randint(2, max_val)
        result.append(x)
        n -= x
    random.shuffle(result)
    return result


def random_integers_sum_to_n_old(n):
    min_length = 3
    max_length = n // 4 + 1
    length = random.randint(min_length, max_length)
    result = []
    remaining_sum = n
    for i in range(length - 1):
        value = random.randint(2, remaining_sum - 2 * (length - i - 1))
        result.append(value)
        remaining_sum -= value
    result.append(remaining_sum)
    random.shuffle(result)
    return result



def random_integers_sum_to_n(n):
    # Calculate minimum and maximum length of the list
    min_length = 3
    max_length = n // 4 + 1
    length = random.randint(min_length, max_length)
    
    # Generate random partition of n into length parts
    partition = [0] * length
    for i in range(n):
        partition[random.randint(0, length-1)] += 1
        
    # Add some randomness to the partition
    for i in range(length):
        partition[i] += random.randint(-1, 1)
        
    # Ensure all integers are at least 2
    for i in range(length):
        if partition[i] < 2:
            partition[i] = 2
            
    # Ensure the sum of the integers is n
    total = sum(partition)
    if total > n:
        diff = total - n
        for i in range(diff):
            partition[random.randint(0, length-1)] -= 1
    elif total < n:
        diff = n - total
        for i in range(diff):
            partition[random.randint(0, length-1)] += 1
            
    random.shuffle(partition)
    return partition


def random_probability_matrix(lp, p):
    # Generate upper triangular matrix of random probabilities
    matrix = [[0] * lp for i in range(lp)]
    for i in range(lp):
        for j in range(i, lp):
            matrix[i][j] = random.uniform(0, p)
            
    # Make the matrix symmetric
    for i in range(lp):
        for j in range(i+1, lp):
            matrix[j][i] = matrix[i][j]
            
    return matrix


def has_self_loops(G):
    for u, v in G.edges():
        if u == v:
            return True
    return False
# n = 30
# m = random.randint(1, n-1)
# G = nx.barabasi_albert_graph(n, m)
# nx.write_gml(G, "testgml.gml")
# quit()


# tau1 = 2
# tau2 = 1.1
# # mu = random.uniform(0.1, 0.9)
# mu = 0.1
# G = nx.LFR_benchmark_graph(10, tau1, tau2, mu, average_degree=3, max_degree=10, min_community=10, max_community=25)

# # print(tau1)
# # print(tau2)
# # print(mu)

# nx.draw(G)
# plt.show()
# quit()

# communities = {frozenset(G.nodes[v]["community"]) for v in G}

# print (communities)

# quit()

# pos = nx.spring_layout(G)

# for node,(x,y) in pos.items():
#     G.nodes[node]['x'] = float(x) * 1000
#     G.nodes[node]['y'] = float(y) * 1000

#nx.write_graphml(G,"..\\Graphs\\test.graphml")
# write_graphml_pos(G, "..\\Graphs\\test.graphml")

# def genrate_lfr_graph(size, seed):
#     params = {"n":size, "tau1":2, "tau2":1.1, "mu":0.1, "min_degree":2, "max_degree":10}
#     # seed = random.randint(1,60)
#     # #seed=10
#     # print(seed)

#     G = nx.LFR_benchmark_graph(params["n"], params["tau1"], params["tau2"], params["mu"], 
#                         min_degree=params["min_degree"],
#                         max_degree=params["max_degree"],
#                         max_iters=5000, seed=seed
#                         )

#     return G  


# worked = []
# yes10 = [3, 10, 20, 43, 53, 93, 94, 130, 143, 177, 189, 190, 193, 196, 221, 249, 275, 278]
# yes10_more = [284, 287, 288, 378, 380, 382, 394, 402, 422, 426, 459, 474, 554, 567, 637, 642, 679, 704, 775, 784]
# yes20 = [3, 53, 130, 143]
# yes70 = [3]
# seeds = range(280,1000)

# for seed in seeds:
#     if len(worked) >= (60 - len(yes10)):
#         break
#     try:
#         G = genrate_lfr_graph(10, seed)
#         worked.append(seed)
#     except:
#         continue
    
#     # nx.draw(G)
#     # plt.show()

#     print(worked)

# yes10.extend(worked)
# print()
# print(worked)
# print(yes10)
# print(len(yes10))
# # for n in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]:

# #     for i in range(60):

# #         G = genrate_lfr_graph(size=n)
        
# #         nx.draw(G)
# #         plt.show()

# quit()

# n = 10
# p_upper = (120 - n) / 500
# p_lower = 1.5 / (n)

# k = random.randint(2, 8)
# p = random.uniform(p_lower+0.1,p_upper+0.2)
# G = nx.newman_watts_strogatz_graph(n, k, p)
# m = G.number_of_edges()
# i = 49
# j = 0
# while m < n or m > 3 * n or not nx.is_connected(G):
#     j += 1
#     if j > 10000:
#         print("nws break due to too many try")
#         break
#     if m < n:
#         #p += 0.005
#         k = random.randint(2, 8)
#         p = random.uniform(p_lower+0.1,p_upper+0.2)
#         G = nx.newman_watts_strogatz_graph(n, k, p)
#         m = G.number_of_edges()
#     elif m > 3 * n:
#         #p -= 0.005
#         k = random.randint(2, 8)
#         p = random.uniform(p_lower+0.1,p_upper+0.2)
#         G = nx.newman_watts_strogatz_graph(n, k, p)
#         m = G.number_of_edges()
#     elif not nx.is_connected(G):
#         #p += 0.005
#         k = random.randint(2, 8)
#         p = random.uniform(p_lower+0.1,p_upper+0.2)
#         G = nx.newman_watts_strogatz_graph(n, k, p)
#         m = G.number_of_edges()
#     else:
#         break
    

# s = "NWS_i{0}_n{1}_m{2}_p{3:.3f}_k{4}.graphml".format(i, n, m, p, k)
# write_graphml_pos(G, s)
# quit()
# graph_list = []

# for i in range(60):
#     G = nx.random_tree(10)
#     G = nx.to_undirected(G)
#     m = G.number_of_edges()

#     is_isomorphic = False
#     for H in graph_list:
#         if nx.is_isomorphic(G, H):
#             is_isomorphic = True
#             break

#     while is_isomorphic:
#         G = nx.random_tree(10)
#         G = nx.to_undirected(G)
#         m = G.number_of_edges()

#         is_isomorphic = False
#         for H in graph_list:
#             if nx.is_isomorphic(G, H):
#                 is_isomorphic = True
#                 break

#     graph_list.append(G)


#     s = "..\\Graphs\\Tree\\TREE_i{0}_n{1}_m{2}.graphml".format(i, 10, m)
#     write_graphml_pos(G, s)



for n in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]:
# for n in [10]:
    p_upper = (120 - n) / 500
    p_lower = 1.5 / (n)
    #print(n, p_upper, p_lower)
    for i in range(500):

        # if i == 20:
            
        #     r = random.randint(61, 10000000)
        #     G = nx.random_tree(n)
        #     G = nx.to_undirected(G)
        #     m = G.number_of_edges()

        #     while not nx.is_connected(G):
        #         r = random.randint(61, 10000)
        #         G = nx.random_tree(n, seed=i+r)
        #         G = nx.to_undirected(G)
        #         m = G.number_of_edges()

        #     # for H in graph_list:
        #     #     while nx.is_isomorphic(G, H):
        #     #         r = random.randint(61, 10000)
        #     #         G = nx.random_tree(n, seed=i+r)
        #     #         G = nx.to_undirected(G)
        #     #         m = G.number_of_edges()
                
        

        #     s = "..\\Graphs\\Tree\\TREE_i{0}_n{1}_m{2}.graphml".format(i, n, m)
        #     write_graphml_pos(G, s)

        # r = random.uniform(0,1)
        # G = nx.random_geometric_graph(n, r)
        # G = nx.to_undirected(G)
        # m = G.number_of_edges()

        # # is_isomorphic = False
        # # for H in graph_list:
        # #     if nx.is_isomorphic(G, H):
        # #         is_isomorphic = True
        # #         break

        # # while is_isomorphic:
        # #     G = nx.random_geometric_graph(n, r)
        # #     G = nx.to_undirected(G)
        # #     m = G.number_of_edges()

        # #     is_isomorphic = False
        # #     for H in graph_list:
        # #         if nx.is_isomorphic(G, H):
        # #             is_isomorphic = True
        # #             break

        # # graph_list.append(G)

        # j = 0
        # while m < n or m > 3 * n or not nx.is_connected(G):
        #     j += 1
        #     if j > 10000:
        #         print(f"{n}:{i} break due to too many try")
        #         break
        #     if m < n:
        #         r = random.uniform(0,1)
        #         G = nx.random_geometric_graph(n, r)
        #         G = nx.to_undirected(G)
        #         m = G.number_of_edges()
        #     elif m > 3 * n:
        #         r = random.uniform(0,1)
        #         G = nx.random_geometric_graph(n, r)
        #         G = nx.to_undirected(G)
        #         m = G.number_of_edges()
        #     elif not nx.is_connected(G):
        #         r = random.uniform(0,1)
        #         G = nx.random_geometric_graph(n, r)
        #         G = nx.to_undirected(G)
        #         m = G.number_of_edges()
        #     else:
        #         break


        # s = "..\\Graphs\\Geometric\\GEO_i{0}_n{1}_m{2}.graphml".format(i, n, m)
        # write_graphml_pos(G, s)
        # continue

        # mn = random.randint(1, 4)
        # G = nx.barabasi_albert_graph(n, mn)
        # m = G.number_of_edges()

        # j = 0
        # while m < n or m > 3 * n or not nx.is_connected(G):
        #     j += 1
        #     if j > 1000:
        #         print(i, "bba break due to too many try")
        #         break
        #     if m < n:
        #         mn += 1
        #         G = nx.barabasi_albert_graph(n, mn)
        #         m = G.number_of_edges()
        #     elif m > 3 * n:
        #         mn -= 1
        #         G = nx.barabasi_albert_graph(n, mn)
        #         m = G.number_of_edges()
        #     elif not nx.is_connected(G):
        #         mn += 1
        #         G = nx.barabasi_albert_graph(n, mn)
        #         m = G.number_of_edges()
        #     else:
        #         break

        # s = "..\\Graphs\\Barabasi-Albert\\BBA_i{0}_n{1}_m{2}.graphml".format(i, n, m)
        # write_graphml_pos(G, s)

        
        p = random.uniform(0, 1)
        G = nx.erdos_renyi_graph(n, p)
        m = G.number_of_edges()

        pos = nx.spring_layout(G)

        for node,(x,y) in pos.items():
            G.nodes[node]['x'] = float(x) * 750
            G.nodes[node]['y'] = float(y) * 750

        # while not nx.is_connected(G):
        #     p = random.uniform(0, 1)
        #     G = nx.erdos_renyi_graph(n, p)
        #     m = G.number_of_edges()

        # j = 0
        # while m < n or m > 3 * n or not nx.is_connected(G):
        #     j += 1
        #     if j > 1000:
        #         print(i, "er break due to too many try")
        #         break
        #     if m < n:
        #         p += 0.005
        #         G = nx.erdos_renyi_graph(n, p)
        #         m = G.number_of_edges()
        #     elif m > 3 * n:
        #         p -= 0.005
        #         G = nx.erdos_renyi_graph(n, p)
        #         m = G.number_of_edges()
        #     elif not nx.is_connected(G):
        #         p += 0.005
        #         G = nx.erdos_renyi_graph(n, p)
        #         m = G.number_of_edges()
        #     else:
        #         break


        s = "..\\Unrestricted_Graphs\\ER_i{0}_n{1}_m{2}_p{3:.3f}.graphml".format(i, n, m, p)
        write_graphml_pos(G, s)

        # k = random.randint(2, 8)
        # p = 0.5
        # G = nx.newman_watts_strogatz_graph(n, k, p)
        # m = G.number_of_edges()

        # j = 0
        # while m < n or m > 3 * n or not nx.is_connected(G):
        #     j += 1
        #     if j > 10000:
        #         print(i, "nws break due to too many try")
        #         break
        #     if m < n:
        #         #p += 0.005
        #         k = random.randint(2, 8)
        #         G = nx.newman_watts_strogatz_graph(n, k, p)
        #         m = G.number_of_edges()
        #     elif m > 3 * n:
        #         #p -= 0.005
        #         k = random.randint(2, 8)
        #         G = nx.newman_watts_strogatz_graph(n, k, p)
        #         m = G.number_of_edges()
        #     elif not nx.is_connected(G):
        #         #p += 0.005
        #         k = random.randint(2, 8)
        #         G = nx.newman_watts_strogatz_graph(n, k, p)
        #         m = G.number_of_edges()
        #     else:
        #         break
            

        # s = "..\\Graphs\\Newman-Watts-Strogatz\\NWS_i{0}_n{1}_m{2}_p{3:.3f}_k{4}.graphml".format(i, n, m, p, k)
        # write_graphml_pos(G, s)

        #l = random.randint(1, n/4)
        # p = random.uniform(0, 1)
        # sizes = random_integers_sum_to_n(n)
        # probs = random_probability_matrix(len(sizes), p)

        
        # G = nx.stochastic_block_model(sizes, probs)
        # G.remove_edges_from(nx.selfloop_edges(G))
        # m = G.number_of_edges()

        # j = 0
        # while m < n or m > 3 * n or not nx.is_connected(G):
        #     j += 1
        #     if j > 10000:
        #         print(i, "sbm break due to too many try")
        #         break

        #     p -= 0.001
        #     if p <= 0:
        #         p = random.uniform(0, 1)

        #     sizes = random_integers_sum_to_n(n)
        #     probs = random_probability_matrix(len(sizes), p)

            
        #     G = nx.stochastic_block_model(sizes, probs)
        #     G.remove_edges_from(nx.selfloop_edges(G))
        #     m = G.number_of_edges()


        # s = "..\\Graphs\\Stochastic-Block-Model\\SBM_i{0}_n{1}_m{2}.graphml".format(i, n, m)
        # write_graphml_pos(G, s)


