

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

from typing import Any, Dict, Type, TypeVar
from dataclasses import dataclass, replace
import json

with open('resources_default.json', 'r') as cfgfile:
    temp_CFG = json.load(cfgfile)
with open('manifest.json', 'r') as f:
    manifest = json.load(f) 
with open('config.json', 'r') as f:
    config = json.load(f)

ALLOWED_CONTAINERS_COUNT = 1
CUBE_FACES = np.array([  # define cube faces using NumPy array reshaping
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [1, 2, 6, 5],
    [0, 3, 7, 4]
])

def init_plt(visual:dict):
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
            func_set_label(visual.get('label_'+axis, 'X-axis'))
            
            # if visual.get('ticks_'+axis, False):
            #     ticks = visual.get('ticks_'+axis)
            #     labels = visual.get('ticklabels_'+axis, [])
            #     if type(ticks) == list:
            #         func_set_ticks(ticks)
            #         if len(labels) == len(ticks): func_set_ticklabels(labels)
            #     if type(ticks) == int and axis in ('x', 'y'):  # z-axis not supported. print warn?.
            #         ax.locator_params(axis=axis, nbins=ticks)
            #         if len(labels) == ticks: func_set_ticklabels(labels)
        
        if visual.get('minorticks', False): ax.minorticks_on()
        if visual.get('showgrid', True): ax.grid()
        ax.set_title(visual.get('label_title', 'Container Visualization'))
        
        return fig, ax

class Unique_ID_Enforcer():
    def __init__(self):
        self.Unique_IDs:set = set()
        match config["header"].get("enforce_no_uid_behaviour"):
            case "replace":
                self.enforce = self._enforce_behaviour_replace
            case _:
                pass
        self._makeObj = self._makeGen()
    def enforce(self):
        raise ValueError("Invalid or Missing Unique ID.")
    def _enforce_behaviour_replace(self):
        return next(self._makeObj)
    def test(self, key, dict) -> int:
        if (key in dict) and (dict[key] not in self.Unique_IDs):
            self.Unique_IDs.add(dict[key])
            return dict[key]
        else:
            self.enforce()
    def _makeGen(self):
        value = 100
        while True:
            if value not in self.Unique_IDs:
                self.Unique_IDs.add(value)
                yield value
            value += 1
uid_e = Unique_ID_Enforcer()

class ResourceHandler():
    def __init__(self):
        pass
    @classmethod
    def _fromfile():
        pass


class Container():
    def __init__(self, data:dict):  # data is expected to be dict holding all Container data.
        self.Container_ID = uid_e.test("Container_ID",data)

        dim:dict = data.get("Dimensions", {})
        self.dimLength = dim.get("Length", 0)
        self.dimWidth = dim.get("Width", 0)
        self.dimHeight = dim.get("Height", 0)

        self.Packages = [Package(pkg) for pkg in data.get("Packages", [])]
    
    @property
    def dim(self):
        return np.array(self.dimLength, self.dimWidth, self.dimHeight)

    def _render(self, fig, ax):
        ax.set_xlim(0, self.dimLength)   # X-axis range
        ax.set_ylim(0, self.dimWidth)   # Y-axis range
        ax.set_zlim(0, self.dimHeight)   # Z-axis range
        for pkg in self.Packages:
            pkg._render(fig, ax)

class CursorHelper():
    def __init__(self):
        self.cursor_state = "highlight" # possible states: highlight, pick
        self.cursor_zpos_real = 0.0  # approximate z-pos of cursor in 3D space, changes when scroll, 
        self.hovered_pkg:set = {}  # a list of currently hovered packages
        self.picked_pkg = None

        self._fixed_view = [None, None, None]
    def cursor_move(self, event):
        if not event.inaxes:
            return
        s = event.inaxes.format_coord(event.xdata, event.ydata)
        s = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", s)
        if float(s[2]) == 0.0: 
            cursorpos = [float(s[0]), float(s[1])] 
        else: return 
        self.cursor_check(cursorpos)
        
    def cursor_click(self, event):
        self.cursor_state = "pick"
        if self.picked_pkg:
            # self._fixed_view[:] = ax.elev, ax.azim, ax.dist
            self.picked_pkg._update_edgecolor("pick")
    
    def cursor_click_release(self, event):
        self.cursor_state = "highlight"
        if self.picked_pkg:
            self.picked_pkg._update_edgecolor("highlight")

    def _highlight_check(self, cursor_pos:list[float]):
        """ Check if the cursor is hovering over any package in 3D space. """
        hovered_pkg_new:set = set()
        for pkg in container.Packages:
            if pkg._cursor_check(cursor_pos):
                hovered_pkg_new.add(pkg)
                if pkg not in self.hovered_pkg:
                    pkg._update_edgecolor("hover")
                    if pkg.posz <= self.cursor_zpos_real <= pkg.posz + pkg.dimHeight and not self.picked_pkg:
                        self.cursor_zpos_real = pkg.posz 
                        self.picked_pkg = pkg
                        pkg._update_edgecolor("highlight")
            else:  # restore
                if pkg in self.hovered_pkg:
                    pkg._update_edgecolor("default")
                if pkg == self.picked_pkg:
                    self.picked_pkg = None
        self.hovered_pkg = hovered_pkg_new

    def _pick_move(self, cursor_pos:list[float]):
        if self.picked_pkg:
            if self._fixed_view[0] is not None:
                # ax.view_init(elev=self._fixed_view[0], azim=self._fixed_view[1])
                # ax.dist = self._fixed_view[2]
                pass
            self.picked_pkg.posx = cursor_pos[0]
            self.picked_pkg.posy = cursor_pos[1]
            self.picked_pkg.polygon.set_verts(
                [self.picked_pkg.pos3d[face] for face in self.picked_pkg._C_FACES]
            )
                
    @property
    def cursor_zpos(self):
        if self.picked_pkg:
            return self.picked_pkg.posz
        return self.cursor_zpos_real
    def cursor_check(self, cursor_pos:list[float]):
        match self.cursor_state:
            case "highlight":
                self._highlight_check(cursor_pos)
            case "pick":
                self._pick_move(cursor_pos)
            case _: raise ValueError("Invalid cursor state.")


class Package():
    _C_FACES = CUBE_FACES
    d_weight = 0.0  # default weight
    d_FaceColors = ['cyan'] * 6  # default face colors
    d_LineWidth = 1  # default line width
    d_EdgeColors = 'r'  # default edge colors
    h_EdgeColors = 'y'  # edge color on hover
    hi_EdgeColors = 'g'  # edge color on highlight
    p_EdgeColors = 'k'  # edge color on pick
    d_Alpha = 0.25  # default alpha
    def __init__(self, data:dict):
        self.Package_ID = uid_e.test("Package_ID",data)

        dim:dict = data.get("Dimensions", {})
        self.dimLength = dim.get("Length", 0)
        self.dimWidth = dim.get("Width", 0)
        self.dimHeight = dim.get("Height", 0)

        self.weight = data.get("Weight", self.d_weight)

        pos:dict = data.get("Position", {})
        self.posx = pos.get("x", 0)
        self.posy = pos.get("y", 0)
        self.posz = pos.get("z", 0)    

        self.FaceColors = self.d_FaceColors
        self.LineWidth = self.d_LineWidth
        self.EdgeColors = self.d_EdgeColors
        self.Alpha = self.d_Alpha

        self._init_polygon()
    
    @property
    def pos(self):
        return np.array(self.posx, self.posy, self.posz)
    @property
    def dim(self):
        return np.array(self.dimLength, self.dimWidth, self.dimHeight)
    @property
    def volume(self):
        return self.dimLength * self.dimWidth * self.dimHeight
    @property
    def pos_midpoint(self):
        return [self.posx + self.dimLength/2,
                self.posy + self.dimWidth/2,
                self.posz + self.dimHeight/2]
    @property
    def pos3d(self):
        x_dim, y_dim, z_dim = self.dimLength, self.dimWidth, self.dimHeight
        x_pos, y_pos, z_pos = self.posx, self.posy, self.posz
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
    
    def _cursor_check(self, cursor_pos:list[float]) -> bool:
        """ Check if the cursor is over the package in 3D space. """
        if self.posx <= cursor_pos[0] <= self.posx + self.dimLength:
            if self.posy <= cursor_pos[1] <= self.posy + self.dimWidth:
                # ignore posz, (current implementation limitation)
                return True
        else: return False

    def _update_edgecolor(self, state:str):
        match state:
            case "hover":
                self.polygon.set_edgecolor(self.h_EdgeColors)
            case "highlight":
                self.polygon.set_edgecolor(self.hi_EdgeColors)
            case "pick":
                self.polygon.set_edgecolor(self.p_EdgeColors)
            case _:
                self.polygon.set_edgecolor(self.EdgeColors)

    def _init_polygon(self):
        self.polygon = Poly3DCollection(
            [self.pos3d[face] for face in self._C_FACES], 
            facecolors=self.FaceColors, 
            linewidths=self.LineWidth, 
            edgecolors=self.EdgeColors, 
            alpha=self.Alpha)

    def _render(self, fig, ax):
        ax.add_collection3d(self.polygon)
        self._update_edgecolor("default")

cur = CursorHelper()
def cursor_move(event):
    if not event.inaxes:
        return
    s = event.inaxes.format_coord(event.xdata, event.ydata)
    s = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", s)
    if float(s[2]) == 0.0: 
        cursorpos = [float(s[0]), float(s[1])] 
    else: return 
    cur.cursor_check(cursorpos)
    fig.canvas.draw_idle()

def mpl_move(event):
    cur.cursor_move(event)
    fig.canvas.draw_idle()

def mpl_click(event):
    cur.cursor_click(event)
    fig.canvas.draw_idle()

def mpl_click_release(event):
    cur.cursor_click_release(event)
    fig.canvas.draw_idle()

def mpl_scroll(event):
    step = config['visual'].get('scroll_step', 0.5)
    if event.button == 'up' and cur.cursor_zpos_real + step <= container.dimHeight:
        cur.cursor_zpos_real += step
    elif event.button == 'down' and cur.cursor_zpos_real - step >= 0:
        cur.cursor_zpos_real -= step
    print(cur.cursor_zpos_real)
    cur.cursor_move(event)
    fig.canvas.draw_idle()

fig, ax = init_plt(config['visual'])
fig.canvas.mpl_connect('motion_notify_event', mpl_move)
toolbar = fig.canvas.manager.toolbar
toolbar.pan(False)
toolbar.zoom(False)
fig.canvas.mpl_connect('button_press_event', mpl_click)
fig.canvas.mpl_connect('button_release_event', mpl_click_release)
fig.canvas.mpl_connect('scroll_event', mpl_scroll)
container = Container(manifest)
container._render(fig, ax)
plt.show()