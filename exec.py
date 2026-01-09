import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from typing import Any, Dict, Type, TypeVar 
from dataclasses import dataclass, replace
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





def generateNewName_incrementSuffix(name:str, nameformat="{name}-child{value}"):
    yield nameformat.format(name=name, value="")
    value = 1
    while True:
        yield nameformat.format(name=name,value=value)
        value += 1


def conv_dims_and_pos_to_3dpos(dim:list[float|int], pos:list[float|int]) -> list[list[float]]:
    """
    Convert dimensions and position to a 3d positions in visualisation.

    dim: tuple of (x_dim, y_dim, z_dim)
    pos: tuple of (x_pos, y_pos, z_pos)
    return: 3d cube vertex array/8 sets of 
    """
    x_dim, y_dim, z_dim = dim
    x_pos, y_pos, z_pos = pos
    array = np.array([
        [x_pos, y_pos, z_pos],                          # origin
        [x_pos + x_dim, y_pos, z_pos],                  
        [x_pos + x_dim, y_pos + y_dim, z_pos],
        [x_pos, y_pos + y_dim, z_pos],
        [x_pos, y_pos, z_pos + z_dim],
        [x_pos + x_dim, y_pos, z_pos + z_dim],
        [x_pos + x_dim, y_pos + y_dim, z_pos + z_dim],
        [x_pos, y_pos + y_dim, z_pos + z_dim]
    ])
    return array

# the important classes
class ContainerHandler():
    """Contains and Handles Container Instances."""
    def __init__(self, name):
        self.name = name
        self.cons:list[Container] = []
        self.pkgs:dict[Package] = {}
        self.selected_con_idx = 0

        # Attached Helper Scripts
        self.con_default_name_generator = generateNewName_incrementSuffix(self.name, nameformat="{name}-container{value}")
    def show(self, fig, ax):
        self._plt_display(fig, ax, self.selected_con_idx)
    @classmethod
    def start_fromfile(cls, filepath):
        data = cls._openfile_json(filepath=filepath)
        src, fig, ax = cls._fromfile(data)
        return src, fig, ax
    @staticmethod
    def _openfile_json(filepath) -> dict[dict[str:str]|list[dict[str:str]]]:
        """Get data from a json file"""
        with open(filepath, 'r') as f:
            data = json.load(f) 
        # add checks here
        return data
    
    @classmethod
    def _fromfile(cls, data) -> tuple[Any]:
        """
        Data will be used to recursively produce multiple layers of classes. 
        Parents will use same data to generate new classes using this common func.
        """
        fig, ax = cls.init_plt(data['vis_header'])
        src = cls(data['vis_header']['label_title'])
        src.pkgs = Package._fromfile(data)
        src.cons = Container._fromfile(data)
        ### call _resolve_ref_at_init to fill _ref for PkgPos Instances.
        for con in src.cons:
            for pp in con.pkgs:
                pp._resolve_ref_at_init(src.pkgs)
        ###
        return src, fig, ax

    def _check_name_valid(self, name):
        if not name: return False
        for con in self.cons:
            if con.name == name: return False
        return True
    
    @property
    def numof_cons(self):
        return len(self.cons)    
    @property
    def numof_pkgs(self):
        return len(self.cons)
    
    
    def _plt_display(self, fig, ax, selected_con_idx=0):
        if len(self.cons) < selected_con_idx:
            raise IndexError('con index out of range'+selected_con_idx)
        print("Displaying ContainerHandler:", self.name)
        ax.set_title(self.name)
        self.cons[selected_con_idx]._plt_display(fig, ax)
        return fig, ax
    def _print_reveal(self):
        """Crude method to display object locations/hierarchy."""
        print("├─SRC    :", self.name)
        for con in self.cons: con._print_reveal()
        for pkg in self.pkgs.values(): pkg._print_reveal()
    def _print_reveal_verbose(self):
        """Cruder method to display object locations/hierarchy"""
        params_list = ['numof_cons', 'numof_pkgs']
        print("├─SRC    :", self.name)
        print("│ └───────┐")
        for p in params_list: print("│         │", p+":", getattr(self,p))
        print("│ ┌───────┘")
        for con in self.cons: con._print_reveal_verbose()
        for pkg in self.pkgs.values(): pkg._print_reveal_verbose()
        print("│")
    
    @staticmethod
    def init_plt(vis_header:dict):
        """
        init routine for the plt module.
        Set up labels, vis environment, window, etc

        Uses the vis-header section.
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

class Container():
    """One individual Container."""
    def __init__(self, name, dim:list[float]):
        self.name = name
        self.dim = dim
        self.pkgs = []  # actually pkg_pos but pp instances SHOULD ref immediately to pkg instances.
        # Attached Helper Scripts
        self.pkg_default_name_generator = generateNewName_incrementSuffix(self.name, nameformat="{name}-package{value}")
    def _check_name_valid(self, name):
        if not name: return False
        for pkg in self.pkgs:
            if pkg.name == name: return False
        return True
    @classmethod
    def _fromfile(cls, data:dict):
        cons = data["con"]
        obj_list = []
        for con in cons:
            obj = cls(con['name'], con['dim']) 
            obj.pkgs = [PackagePosition(ref=pp['name'], pos=pp['pos'], dim=pp['dim']) for pp in con['pkg_pos']]
            obj_list.append(obj)
        return obj_list
    @property
    def numof_pkgs(self):
        return len(self.pkgs)
    def _print_reveal(self):
        """Crude method to display object locations."""
        print("│ ├─CON  :", self.name)
        for pkg in self.pkgs: pkg._print_reveal()
    def _print_reveal_verbose(self):
        """ Even Cruder method to display object location and its hierarchy"""
        params_list = ('dim', "numof_pkgs")
        print("│ ├─CON  :", self.name)
        print("│ │ └─────┐")
        for p in params_list: print("│ │       │", p+":", getattr(self,p))
        print("│ │ ┌─────┘")
        for pkg in self.pkgs: pkg._print_reveal_verbose()
        print("│ │")
    def _plt_display(self, fig, ax):
        """Add container data into fig, ax."""
        print("Displaying Container:", self.name)
        ax.set_xlim([0, self.dim[0]])
        ax.set_ylim([0, self.dim[1]])
        ax.set_zlim([0, self.dim[2]])
        for pkg in self.pkgs:
            pkg._plt_display(fig, ax)


class PackageResource():
    """Holds data resources for pkg instances."""
    resource_type = "parent"
    name: str
    CFG: dict = {}
    def __init__(self, name):
        self.name = name  # name of the data resource
    
    @classmethod
    def _fromfile(cls, data:dict):
        data = cls._apply_default(data)
        return cls(**data)
    
    def _tofile(self) -> dict:
        raise NotImplementedError()
    
    def _edit(self, **changes):
        """Resources cannot be edited. Changes must take place by creating a new resource instance."""
        if "name" in changes and changes["name"]==self.name: changes["name"] = "copy: "+changes["name"]  #ensure a new name every time. TODO replace this ith something better
        if changes:
            return replace(self, **changes)
    
    @classmethod
    def _apply_default(cls, data:dict) -> dict:
        for name, default in cls.CFG.items():
            if name not in data or data[name] in (None, "", [], {}):
                data[name] = default
        return data
    
class ColourResource(PackageResource):
    resource_type = "colours"
    name: str
    fillcolour: str; edgecolour: str; alpha: float; linewidth: float
    CFG = {"fillcolour": "magenta", "edgecolour": "red", "alpha": 0.6, "linewidth": 1.} # TODO get from .cfg
    def __init__(self, name, fillcolour: str, edgecolour: str, alpha: float, linewidth: float):
        super().__init__(name)
        self.fillcolour = fillcolour; self.edgecolour = edgecolour
        self.alpha = alpha; self.linewidth = linewidth


DEFAULT_RESOURCE = {
    'colour': ColourResource._fromfile({"name":"default"})}

class Package():
    """One individual package."""
    def __init__(self, name, colour=DEFAULT_RESOURCE['colour']):
        self.name = name
        self.colour = colour

    @classmethod
    def _fromfile(cls, data:dict):
        pkgs = data['pkg']  # get templates from here
        obj = {k: cls(v['name']) for k, v in pkgs.items()}
        return obj

    def __getattr__(self, p):
        return getattr(self.colour, p)

    def _print_reveal(self):
        """Crude method to display obj locations."""
        print("│ ├─PKG:", self.name)
    def _print_reveal_verbose(self):
        """ Even cruder method to display object location """
        self._print_reveal()
        # params_list = ('dim', "pos", "pos_midpoint")
        # print("│ │ ├─PKG:", self.name)
        # print("│ │ │ └───┐")
        # for p in params_list: print("│ │ │     │ ", p+":", getattr(self,p))
        # TODO update

class PackagePosition():
    _C_FACES = CUBE_FACES
    def __init__(self, pos, dim, ref:Package|str):
        self.pos = pos
        self.dim = dim
        self._ref = ref
    @property
    def _3dpos(self):
        return conv_dims_and_pos_to_3dpos(self.dim, self.pos)
    @property
    def pos_midpoint(self):
        return [self.pos[0] + self.dim[0]/2,
                self.pos[1] + self.dim[1]/2,
                self.pos[2] + self.dim[2]/2]
    @property
    def pkg(self):
        return self._ref
    
    def __getattr__(self, p):
        return getattr(self._ref, p)

    def _resolve_ref_at_init(self, pkgs:dict[str:Package]):  # yuck! not pythonic
        # i dug myself in a corner here, ConHandler not inited so: ref to Pkg needs direct connection. 
        if self._ref in pkgs:
            self._ref = pkgs[self._ref]
    def _plt_display(self, fig, ax):
        ax.add_collection3d(Poly3DCollection([self._3dpos[face] for face in self._C_FACES], facecolors=self.fillcolour, linewidths=self.linewidth, edgecolors=self.edgecolour, alpha=self.alpha))

    def _print_reveal(self):
        pass
    def _print_reveal_verbose(self):
        print("│ │ ├─"+self.name, self.pos, f"{self.dim[0]}x{self.dim[1]}x{self.dim[2]}")

verify_data()
src, fig, ax = ContainerHandler.start_fromfile("data.json")
src._print_reveal_verbose()
src.show(fig, ax)
plt.show()
