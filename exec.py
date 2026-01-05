import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json

ALLOWED_CONTAINERS_COUNT = 1
CUBE_FACES = np.array([  # define cube faces using NumPy array reshaping
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [1, 2, 6, 5],
    [0, 3, 7, 4]
])

# read the json file
with open('data.json', 'r') as f:
    data:dict[dict[str:str]|list[dict[str:str]]] = json.load(f) 
    header:dict[str:str] = data['header']  # private data
    vis_header:dict[str:str] = data['vis_header']  # user-editable data
    con:list[dict[str:str]] = data['con']  # container info
    pkg:list[dict[str:str]] = data['pkg']  # package info
    meta:dict[str:str] = data['meta']  # related metadata
    MEASURE_UNIT = vis_header.get('measure-unit', 'mm')

def verify_data():
    """
    Verify that the data meets the required conditions.
    """
    if len(con) != header.get('con_count', 0):
        raise ValueError("Container count error. (does not match header)")
    if len(pkg) != header.get('pkg_count', 0):
        raise ValueError("Package count error. (does not match header)")
    if len(con) > ALLOWED_CONTAINERS_COUNT or header.get('con_count', 0) > ALLOWED_CONTAINERS_COUNT:
        raise ValueError(f"Number of containers must be {ALLOWED_CONTAINERS_COUNT}.")

    # Additional verification logic can be added here

def init_plt():
    """
    Basic init routine for the plt module.
    Set up labels, vis environment, window

    DONT PUT TESTS HERE
    NOT CON, NOT PKG EITHER.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    ax.set_autoscale_on(False)

    for axis, func_set_label, func_set_ticks, func_set_ticklabels in zip(['x', 'y', 'z'], [ax.set_xlabel, ax.set_ylabel, ax.set_zlabel], [ax.set_xticks, ax.set_yticks, ax.set_zticks], [ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels]):
        func_set_label(vis_header.get('label_'+axis, 'X-axis')+f" ({MEASURE_UNIT})")
        
        if vis_header.get('ticks_'+axis, False):
            ticks = vis_header.get('ticks_'+axis)
            labels = vis_header.get('ticklabels_'+axis, [])
            if type(ticks) == list:
                func_set_ticks(ticks)
                if len(labels) == len(ticks): func_set_ticklabels(labels)
            if type(ticks) == int and axis in ('x', 'y'):  # z-axis not supported. print warn?.
                ax.locator_params(axis=axis, nbins=ticks)
                if len(labels) == ticks: func_set_ticklabels(labels)
    
    if vis_header.get('minorticks', False): ax.minorticks_on()
    if vis_header.get('showgrid', True): ax.grid()
    ax.set_title(vis_header.get('label_title', 'Container Visualization'))
    
    return fig, ax


def display_con(fig, ax, con):
    """
    Visualise the container. The container is represented by the axes themselves.

    fig, ax: plt figure and axes
    con: container data
    """
    shape = tuple(con.get('dims', (5, 5, 5)))
    ax.set_xlim([0, shape[0]])
    ax.set_ylim([0, shape[1]])
    ax.set_zlim([0, shape[2]])

def conv_dims_and_pos_to_cubevertices(dims, pos):
    """
    Convert dimensions and position to a voxel array.

    dims: tuple of (x_dim, y_dim, z_dim)
    pos: tuple of (x_pos, y_pos, z_pos)
    return: 3d cube vertex array
    """
    x_dim, y_dim, z_dim = dims
    x_pos, y_pos, z_pos = pos
    array = np.array([
        [x_pos, y_pos, z_pos],
        [x_pos + x_dim, y_pos, z_pos],
        [x_pos + x_dim, y_pos + y_dim, z_pos],
        [x_pos, y_pos + y_dim, z_pos],
        [x_pos, y_pos, z_pos + z_dim],
        [x_pos + x_dim, y_pos, z_pos + z_dim],
        [x_pos + x_dim, y_pos + y_dim, z_pos + z_dim],
        [x_pos, y_pos + y_dim, z_pos + z_dim]
    ])
    return array

def display_pkg(fig, ax, pkg:dict):
    """
    Visualise a package as a cuboid in the 3D plot.

    fig, ax: plt figure and axes
    pkg: package data
    """
    dims = tuple(pkg.get('dims', (1, 1, 1)))
    pos = tuple(pkg.get('pos', (0, 0, 0)))
    colour = pkg.get('colour', {'fill':'purple', 'edge':'red', 'alpha':.6})
    vertices = conv_dims_and_pos_to_cubevertices(dims, pos)
    ax.add_collection3d(
        Poly3DCollection(
            [vertices[CUBE_FACES[face]] for face in range(len(CUBE_FACES))],
            facecolors=colour['fill'], linewidths=1, edgecolors=colour['edge'], alpha=colour['alpha']
        )
    )

verify_data()
fig, ax = init_plt()
display_con(fig, ax, con[0])
for pkg in pkg:
    display_pkg(fig, ax, pkg)
plt.show()