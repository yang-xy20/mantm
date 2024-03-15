import sys
import matplotlib
import numpy as np  
# if sys.platform == 'darwin':
# matplotlib.use("tkagg")
# else:
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as lines 
import seaborn as sns
import skimage

def draw_pose(ax, pos, grid, color="Grey", agent_size=8, alpha=0.9):
    x, y, o = pos
    x, y = x * 100.0 / 5.0, grid.shape[1] - y * 100.0 / 5.0

    dx = 0
    dy = 0
    fc = color
    dx = np.cos(np.deg2rad(o))
    dy = -np.sin(np.deg2rad(o))
    ax.arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * (agent_size * 1.25),
                head_width=agent_size, head_length=agent_size * 1.25,
                length_includes_head=True, fc=fc, ec=fc, alpha=alpha)
    
def draw_circle(ax, pos, grid, color="Grey", alpha=0.9, radius=3):
    try:
        (x, y, _)= pos[0]
    except:
        x, y = pos[1]/100*5, pos[0]/100*5
    x, y = x * 100.0 / 5.0, grid.shape[1] - y * 100.0 / 5.0
    cir1 = Circle(xy = (x, y), radius=radius, alpha=alpha, fc=color, ec=color)
    ax.add_patch(cir1)
    
    
def draw_line(ax, node_pos, neighbor_pos, grid, color="Grey", alpha=0.9):
    (node_x, node_y, _)= node_pos
    node_x, node_y = node_x * 100.0 / 5.0, grid.shape[1] - node_y * 100.0 / 5.0
    
    (neighbor_x, neighbor_y, _)= neighbor_pos
    neighbor_x, neighbor_y = neighbor_x * 100.0 / 5.0, grid.shape[1] - neighbor_y * 100.0 / 5.0
    
    line = lines.Line2D([node_x, neighbor_x], 
                        [node_y, neighbor_y], 
                        lw = 0.5, color =color, 
                        axes = ax, alpha = alpha) 
    ax.add_line(line) 


def visualize_all(agent_id, fig, ax, img, grid_gt, pos, pos_gt, node_list, affinity, dump_dir, t,
              visualize, save_gifs):
    
    for i in range(2):
        ax[i].clear()
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])

    #########################obs###########################
    ax[0].imshow(img)
    ax[0].set_title("Agent {} Obs.".format(agent_id), family='sans-serif',
                    fontname='DejaVu Sans',
                    fontsize=15)
    ##########################ground-truth map###########################
    title = "Agent {} Local Map".format(agent_id)
    
    ax[1].imshow(grid_gt)
    ax[1].set_title(title, family='sans-serif',
                    fontname='DejaVu Sans',
                    fontsize=15)

    # Draw GT agent pose
    draw_pose(ax[1], pos_gt, grid_gt, color="Grey", agent_size=8, alpha=0.9)

    # Draw predicted agent pose
    draw_pose(ax[1], pos, grid_gt, color="Red", agent_size=8, alpha=0.9)
    if np.all(node_list) != None and np.all(affinity) != None:
        draw_point_list = []
        for idx, node_position in enumerate(node_list):
            neighbors = np.where(affinity[idx])[0]
            for neighbor_idx in neighbors:
                neighbor_position = node_list[neighbor_idx]
                draw_line(ax[1], node_position, neighbor_position, grid_gt, color="blue", alpha=0.9)
            draw_point_list.append([node_position])

        for node_position in draw_point_list:
            draw_circle(ax[1], node_position, grid_gt, color="blue", alpha=0.9)
        #self._draw_boundary(self.node_list[curr_info['curr_node']], CURR_NODE)


    # for _ in range(5):
    #     fig.tight_layout()

    if visualize:
        plt.gcf().canvas.flush_events()
        fig.canvas.start_event_loop(0.001)
        plt.gcf().canvas.flush_events()
    
    # plt.show()

    if save_gifs:
        fn = '{}/step-{:0>4d}.png'.format(dump_dir, t)
        fig.savefig(fn)

def visualize_frontier_map(fig, ax, grid_gt, grid_pred, pos_gt, pos_pred, all_global_goal, dump_dir, t, visualize, save_gifs, use_local):
    
    global_goal = (all_global_goal[-2:])
    all_goal = (all_global_goal[:-2])
    ax.clear()
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    title = ""

    ax.imshow(grid_pred)
    ax.set_title(title, family='sans-serif',
                    fontname='DejaVu Sans',
                    fontsize=20)

    agent = 0
    for p_gt, p_pred in zip(pos_gt, pos_pred):
        # Draw GT agent pose
        if agent%2==0:
            draw_pose(ax, p_gt, grid_pred, color="Grey", agent_size=8, alpha=0.9)

        # Draw predicted agent pose
        else:
            draw_pose(ax, p_gt, grid_pred, color="Red", agent_size=8, alpha=0.9)
        #draw_pose(ax, p_pred, grid_pred, color="blue", agent_size=8, alpha=0.6)
        agent += 1
    agent = 0
    for pos in (global_goal):
        if pos[0]==0 and pos[1]==0:
            break
        if agent % 2 ==0:
            draw_circle(ax, pos, grid_gt, color="Grey", alpha=0.9)
        else:
            draw_circle(ax, pos, grid_gt, color="Red", alpha=0.9)
        agent += 1
    for goal in all_goal:
        invalid = False
        if use_local:
            for pos_a in pos_gt:
                y, x = pos_a[0]*100/5, pos_a[1]*100/5
                dis = ((goal[0] - x)**2 + (goal[1] - y)**2)**0.5
                if dis > 120:
                    invalid = True
                    break
        if invalid:
            draw_circle(ax, goal, grid_gt, color="Red", alpha=0.9, radius=1)
        else:
            draw_circle(ax, goal, grid_gt, color="blue", alpha=0.9, radius=1)

    for _ in range(5):
        fig.tight_layout()

    if visualize:
        plt.gcf().canvas.flush_events()
        fig.canvas.start_event_loop(0.001)
        plt.gcf().canvas.flush_events()

    if save_gifs:
        fn = '{}/step-{:0>4d}.png'.format(dump_dir, t)
        fig.savefig(fn)

def visualize_map(fig, ax, grid_gt, grid_pred, pos_gt, pos_pred, merge_node_list, merge_affinity, global_goal, merge_ghost_node, merge_ghost_mask, dump_dir, t, visualize, save_gifs):
    
    ax.clear()
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    title = ""

    ax.imshow(grid_pred)
    ax.set_title(title, family='sans-serif',
                    fontname='DejaVu Sans',
                    fontsize=20)

    for p_gt, p_pred in zip(pos_gt, pos_pred):
        # Draw GT agent pose
        draw_pose(ax, p_gt, grid_pred, color="Grey", agent_size=8, alpha=0.9)

        # Draw predicted agent pose
        draw_pose(ax, p_pred, grid_pred, color="Red", agent_size=8, alpha=0.6)
    
    if np.all(merge_node_list) != None and np.all(merge_affinity) != None:
        draw_point_list = []
        for idx, node_position in enumerate(merge_node_list):
            neighbors = np.where(merge_affinity[idx])[0]
            for neighbor_idx in neighbors:
                neighbor_position = merge_node_list[neighbor_idx]
                draw_line(ax, node_position, neighbor_position, grid_gt, color="blue", alpha=0.9)
            draw_point_list.append([node_position])
        for idx, node_position in enumerate(draw_point_list):
            if idx in global_goal:
                draw_circle(ax, node_position, grid_gt, color="red", alpha=0.9)
            else:
                draw_circle(ax, node_position, grid_gt, color="blue", alpha=0.9)
        #self._draw_boundary(self.node_list[curr_info['curr_node']], CURR_NODE)
        
    if np.all(merge_ghost_node) != None and np.all(merge_ghost_mask) != None:
        draw_point_list = []
        for idx in range(merge_ghost_node.shape[0]):
            for idy in range(merge_ghost_node.shape[1]):
                if merge_ghost_mask[idx, idy] == 0:
                    continue
                else:
                    node_position = merge_ghost_node[idx, idy]
                    neighbor_position = merge_node_list[idx]
                    draw_line(ax, node_position, neighbor_position, grid_gt, color="grey", alpha=0.9)
                    draw_point_list.append([node_position])
        for node_position in draw_point_list:
            draw_circle(ax, node_position, grid_gt, color="grey", alpha=0.9)
        # for i in range(frontier_loc.shape[0]):
        #     draw_circle(ax, [frontier_loc[i]*5/100.0], grid_gt, color="blue", alpha=0.9)
        #self._draw_boundary(self.node_list[curr_info['curr_node']], CURR_NODE)


    for _ in range(5):
        fig.tight_layout()

    if visualize:
        plt.gcf().canvas.flush_events()
        fig.canvas.start_event_loop(0.001)
        plt.gcf().canvas.flush_events()

    if save_gifs:
        fn = '{}/step-{:0>4d}.png'.format(dump_dir, t)
        fig.savefig(fn)

'''
def visualize_map(fig, ax, grid_gt, grid_pred, pos_gt, pos_pred, dump_dir, t, visualize, save_gifs):
    
    for i in range(2):
        ax[i].clear()
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])

    title = "Merged Pred. Map and Pose"

    ax[0].imshow(grid_pred)
    ax[0].set_title(title, family='sans-serif',
                    fontname='DejaVu Sans',
                    fontsize=20)

    for p_gt, p_pred in zip(pos_gt, pos_pred):
        # Draw GT agent pose
        draw_pose(ax[0], p_gt, grid_pred, color="Grey", agent_size=8, alpha=0.9)

        # Draw predicted agent pose
        draw_pose(ax[0], p_pred, grid_pred, color="Red", agent_size=8, alpha=0.6)

    title = "Merged GT Map and Pose"

    ax[1].imshow(grid_gt)
    ax[1].set_title(title, family='sans-serif',
                    fontname='DejaVu Sans',
                    fontsize=20)

    for p_gt, p_pred in zip(pos_gt, pos_pred):
        # Draw GT agent pose
        draw_pose(ax[1], p_gt, grid_gt, color="Grey", agent_size=8, alpha=0.9)

        # Draw predicted agent pose
        draw_pose(ax[1], p_pred, grid_gt, color="Red", agent_size=8, alpha=0.6)

    for _ in range(5):
        fig.tight_layout()

    if visualize:
        plt.gcf().canvas.flush_events()
        fig.canvas.start_event_loop(0.001)
        plt.gcf().canvas.flush_events()

    if save_gifs:
        fn = '{}/step-{:0>4d}.png'.format(dump_dir, t)
        fig.savefig(fn)'''

def insert_circle(mat, x, y, value):
    mat[x - 2: x + 3, y - 2:y + 3] = value
    mat[x - 3:x + 4, y - 1:y + 2] = value
    mat[x - 1:x + 2, y - 3:y + 4] = value
    return mat

def fill_color(colored, mat, color):
    for i in range(3):
        colored[:, :, 2 - i] *= (1 - mat)
        colored[:, :, 2 - i] += (1 - color[i]) * mat
    return colored

def get_colored_map(mat, collision_map, visited, visited_gt, goal,
                    explored, gt_map, gt_map_explored):
    m, n = mat.shape
    colored = np.zeros((m, n, 3))
    pal = sns.color_palette("Paired")

    current_palette = [(0.9, 0.9, 0.9)]
    colored = fill_color(colored, gt_map, current_palette[0]) # gray

    current_palette = [(235. / 255., 243. / 255., 1.)]
    colored = fill_color(colored, explored, current_palette[0]) # sky blue

    green_palette = sns.light_palette("green")
    colored = fill_color(colored, mat, pal[2]) # light green 

    current_palette = [(0.6, 0.6, 0.6)]
    colored = fill_color(colored, gt_map_explored, current_palette[0]) #gray

    colored = fill_color(colored, mat * gt_map_explored, pal[3]) # dark green

    red_palette = sns.light_palette("red")

    colored = fill_color(colored, visited_gt, current_palette[0]) # gray
    colored = fill_color(colored, visited, pal[4]) # pink
    colored = fill_color(colored, visited * visited_gt, pal[5]) # red

    colored = fill_color(colored, collision_map, pal[2]) # light green 

    current_palette = sns.color_palette()

    selem = skimage.morphology.disk(4)
    for g in goal:
        goal_mat = np.zeros((m, n))
        goal_mat[g[0], g[1]] = 1
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal_mat, selem) != True

        colored = fill_color(colored, goal_mat, current_palette[0])

    current_palette = sns.color_palette("Paired")

    colored = 1 - colored
    colored *= 255
    colored = colored.astype(np.uint8)
    return colored
