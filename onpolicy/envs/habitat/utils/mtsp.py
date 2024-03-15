from . import pose as pu
from .kmeans import KMeans
import numpy as np

def compute_frontier_agent(frontier_pos, agent_pos):
    cluster = [[] for _ in range(self.num_agents)]
    for i in range(frontier_pos):
        dis = np.inf
        for j in range(agent_pos):
            f_r_dis = pu.get_l2_distance(frontier_pos[i][0], agent_pos[j][0], frontier_pos[i][1], agent_pos[j][1])
            if f_r_dis < dis:
                dis = f_r_dis
                index = j
        cluster[index].append(frontier_pos[i])
    return cluster

def compute_frontier_plan(frontier_cluster):
    frontier_matrix = np.ones((len(frontier_cluster),len(frontier_cluster)))*1000
    for i in range(frontier_matrix.shape[0]):
        for j in range(frontier_matrix.shape[1]):
            if i != j:
                frontier_matrix[i,j] = pu.get_l2_distance(frontier_cluster[i][0], frontier_cluster[j][0], \
                frontier_cluster[i][1], frontier_cluster[j][1])
    min_index = np.argmin(frontier_matrix[0])
    frontier_matrix[:, min_index] = 1000
    index = [min_index]
    for i in range(frontier_matrix.shape[0]-1):
        min_index = np.argmin(frontier_matrix[min_index])
        frontier_matrix[:, min_index] = 1000
        index.append(min_index)
    return index

def compute_kmeans(data, k=2, init_centers=None, init_method="random"):
    agent_frontier_node = [0 for _ in range(k)]
    kmeans = KMeans(data, k=k, init_centers=init_centers, init_method=init_method)
    while True:
        kmeans.step()
        if kmeans.check_end():
            break
    for i in range(k):
        selected_data = kmeans.fetch_data(i)
        agent_frontier_node[i] = selected_data
    
    return agent_frontier_node, kmeans.centers[-1]

    