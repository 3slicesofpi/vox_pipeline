import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

import json

with open('manifest.json', 'r') as f:
    manifest = json.load(f) 
with open('setup/config.json', 'r') as f:
    config = json.load(f)
with open('setup/typedata_container.json', 'r') as f:
    td_con:dict[dict] = json.load(f)
with open('setup/typedata_package.json', 'r') as f:
    td_pkg:dict[dict] = json.load(f)

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
        toolbar = fig.canvas.manager.toolbar
        
        ax.set_xlabel(visual.get('label_x', "Length"))
        ax.set_ylabel(visual.get('label_y', "Width"))
        ax.set_zlabel(visual.get('label_z', "Height"))
        
        if visual.get('minorticks', False): ax.minorticks_on()
        if visual.get('showgrid', True): ax.grid()
        ax.set_title(visual.get('label_title', 'Container Visualization'))
        toolbar.pan(False)
        toolbar.zoom(False)
        toolbar.mode = None
        return fig, ax, toolbar

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

class New_Type_Generator():
    def __init__(self):
        self._makeObj = self._makeGen()
    def new(self):
        return next(self._makeObj)
    def _makeGen(self):
        while True:
            yield "ErrorType"
ntype_g = New_Type_Generator()

class ResourceHandler():
    def __init__(self):
        pass
    @classmethod
    def _fromfile():
        pass


class Container():
    def __init__(self, data:dict):  # data is expected to be dict holding all Container data.
        self.Container_ID = uid_e.test("Container_ID",data)
        self.Container_Type = data.get("Container_Type", ntype_g.new())
        td:dict = td_con.get(self.Container_Type, "Default")

        dim:dict = td.get("Dimensions", {})
        self.dimLength = dim.get("Length", 0)
        self.dimWidth = dim.get("Width", 0)
        self.dimHeight = dim.get("Height", 0)

        self.weight = td.get("Weight", 0)

        self.Packages = [Package(pkg) for pkg in data.get("Packages", [])]
    @property
    def volume(self):
        return self.dimLength * self.dimWidth * self.dimHeight
    @property
    def dim(self):
        return np.array(self.dimLength, self.dimWidth, self.dimHeight)

    def _render(self, fig, ax):
        ax.set_xlim(0, self.dimLength)   # X-axis range
        ax.set_ylim(0, self.dimWidth)   # Y-axis range
        ax.set_zlim(0, self.dimHeight)   # Z-axis range
        # If dimLength, dimWidth, dimHeight are not equal, set_box_aspect to avoid distortion
        # However, if they differ too much, print a warning.
        if not (self.dimLength == self.dimWidth == self.dimHeight):
            ratio_max = max(self.dimLength, self.dimWidth, self.dimHeight) / min(self.dimLength, self.dimWidth, self.dimHeight)
            if ratio_max > 5:
                print(f"Warning: Container dimensions differ significantly (Ratio: {ratio_max:.2f}). Visualization may be distorted.")
            ax.set_box_aspect([self.dimLength, self.dimWidth, self.dimHeight])
        for pkg in self.Packages:
            pkg._render(fig, ax)
    
    def _export(self) -> dict:
        volume_amt = sum(p.volume for p in self.Packages)
            
        return {
            "Container_ID" : self.Container_ID,
            "Container_Type": self.Container_Type,
            "Dimensions" : {"Length": self.dimLength, "Width": self.dimWidth, "Height": self.dimHeight},
            "Weight" : self.weight,
            "Gross_Weight" : self.weight+sum(p.weight for p in self.Packages),
            "Volume_Used_Amount" : volume_amt,
            "Volume_Utilization":  f"{volume_amt/self.volume:.4f}",
            "Package_Slate": self.collate_Package_Slate(),
            "Packages" : [p._export() for p in self.Packages]
        }    

    def collate_Package_Slate(self) -> dict:
        slate = {}
        for pkg in self.Packages:
            if pkg.Package_Type in slate:
                slate[pkg.Package_Type]["Quantity"] += 1
            else:
                td_pkg_i = pkg._export()
                slate[pkg.Package_Type] = {
                    "Package_Type" : td_pkg_i["Package_Type"],
                    "Quantity" : 1,
                    "Dimensions" : td_pkg_i["Dimensions"],
                    "Weight" : td_pkg_i["Weight"]
                }
        return slate

    
    def export_tofile(self, filename:str="export.json"):
        with open(filename, 'w') as f:
            json.dump(self._export(), f, indent=2)
        f.close()

class Package():
    _C_FACES = CUBE_FACES
    d_weight = 0.0  # default weight
    d_FaceColors = ['grey'] * 6  # default face colors
    d_LineWidth = 1  # default line width
    d_EdgeColors = 'r'  # default edge colors
    h_EdgeColors = 'y'  # edge color on hover
    f_EdgeColors = 'g'  # edge color on focus
    p_EdgeColors = 'k'  # edge color on pick
    d_Alpha = 0.25  # default alpha
    def __init__(self, data:dict):
        self.Package_ID = uid_e.test("Package_ID",data)
        self.Package_Type = data.get("Package_Type", ntype_g.new())
        td:dict = td_pkg.get(self.Package_Type, "Default")

        dim:dict = td.get("Dimensions", {})
        self.dimLength = dim.get("Length", 0)
        self.dimWidth = dim.get("Width", 0)
        self.dimHeight = dim.get("Height", 0)

        self.weight = td.get("Weight", self.d_weight)

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

    def _cursor_check(self, posx, posy, posz) -> 0|1|2:
        """ Check if the cursor is over the package in 3D space. """
        if self.posx <= posx <= self.posx + self.dimLength:
            if self.posy <= posy <= self.posy + self.dimWidth:
                if self.posz <= posz <= self.posz + self.dimHeight:
                    return 2
                return 1
            return 0
        return 0

    def _update_edgecolor(self, state:str):
        match state:
            case "hover":
                self.polygon.set_edgecolor(self.h_EdgeColors)
            case "focus":
                self.polygon.set_edgecolor(self.f_EdgeColors)
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

    def update_pos(self):
        self.polygon.set_verts(
                    [self.pos3d[face] for face in self._C_FACES]
                )

    def _render(self, fig, ax):
        ax.add_collection3d(self.polygon)
        self._update_edgecolor("default")

    def _export(self):
        return {
            "Package_ID" : self.Package_ID,
            "Package_Type": self.Package_Type,
            "Dimensions" : {"Length" : self.dimLength, "Width" : self.dimWidth, "Height" : self.dimHeight},
            "Weight" : self.weight,
            "Position": {"x": self.posx, "y":self.posy, "z":self.posz}
        }

class CursorHelper():
    def __init__(self):
        self.posx = 0.0
        self.posy = 0.0
        self.posz = 0.0

        self.onfloor = False
        self.state = "hover" # point hover pick
        self.hoverpkg = []
        self.focuspkg = None

        (self.probeline,) = ax.plot(
            [0, 0], [0, 0], [0, ax.get_zlim()[1]],
            color="black", linewidth=1.5
            ) 

    def _update(self):
        for pkg in container.Packages:
            pkg._update_edgecolor("default")
        for pkg in self.hoverpkg:
            pkg._update_edgecolor("hover")
        if self.focuspkg:
            self.focuspkg._update_edgecolor("focus")
        fig.canvas.draw_idle()

    def _mpl_move(self, event):
        if not event.inaxes:
            self.probeline.set_visible(False)
            self._update()
            return 
        s:str = event.inaxes.format_coord(event.xdata, event.ydata)
        s = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", s)
        if float(s[2]) <= 0.001: 
            self.posx = float(s[0])
            self.posy = float(s[1])
            self.onfloor = True
        else: 
            self.onfloor = False
            self.probeline.set_visible(False)
            self._update()
            return
        self.probeline.set_data([self.posx, self.posx], [self.posy, self.posy])
        self.probeline.set_3d_properties([0, container.dimHeight])
        self.probeline.set_visible(True)
        print(self.posx, self.posy, self.posz, self.state)

        if self.state == "hover":
            for pkg in container.Packages:
                match pkg._cursor_check(self.posx, self.posy, self.posz):
                    case 2: 
                        self.focuspkg = pkg
                        if pkg not in self.hoverpkg:
                            self.hoverpkg.append(pkg)
                    case 1:
                        if pkg not in self.hoverpkg:
                            self.hoverpkg.append(pkg)
                    case 0:
                        if pkg in self.hoverpkg:
                            self.hoverpkg.remove(pkg)
                        if pkg == self.focuspkg:
                            self.focuspkg = None

            self._update()
        elif self.state == "focus":
            # Move focused package
            if self.focuspkg:
                self.focuspkg.posx = self.posx
                self.focuspkg.posy = self.posy
                self.focuspkg.update_pos()
                self._update()

    def _mpl_click(self, event):
        if not event.inaxes:
            return
        elif self.focuspkg is not None and self.state == "hover" and event.button==3:
            self.focuspkg._update_edgecolor("pick")
            self.state = "focus"
            self._update()
    
    def _mpl_release(self, event):
        self.state = "hover"
        self.focuspkg = None
        self.hoverpkg = []
                
    def _mpl_scroll(self, event):
        if not event.inaxes and self.onfloor:
            return
        match self.state:
            case "click": return
            case "hover":
                zposes = {pkg.posz: pkg for pkg in self.hoverpkg}
                zposes = dict(sorted(zposes.items()))
                print(zposes)
                if event.button == 'up':  # bring self.posz to the next highest value on zposes.
                    for z, pkg in zposes.items():
                        if z > self.posz and z != self.posz:
                            self.posz = z
                            self.focuspkg = pkg
                            self._update(); return
                else:  # bring self.posz to the next lowest value on zposes.
                    for z, pkg in dict(reversed(list(zposes.items()))).items():
                        if z < self.posz and z != self.posz:
                            self.posz = z
                            self.focuspkg = pkg
                            self._update(); return
            case "focus":
                if event.button == 'up' and self.posz + 0.1 <= container.dimHeight - self.focuspkg.dimHeight:
                    self.posz += 0.1; self.focuspkg.posz += 0.1
                    self.focuspkg.update_pos()
                    self._update(); return
                elif event.button == 'down' and self.posz - 0.1 >= 0:
                    self.posz -= 0.1; self.focuspkg.posz -= 0.1
                    self.focuspkg.update_pos()
                    self._update(); return


def mpl_onkey(event):
    match event.key:
        case 'ctrl+s':
            print("Info: To save edited manifest, press [ALT]+[S].")
        case 'alt+s':
            print("Info: Saving new manifest...")
            container.export_tofile()
            print("Info: Saving Load Views... ", end='')
            ax.xaxis.pane.set_visible(False)
            ax.yaxis.pane.set_visible(False)
            ax.zaxis.pane.set_visible(False)
            for name, view in {
                #name: (elev, azim, proj)
                "iso":   (30, -60, "ortho"),
                "top":   (90, -90, "persp"),
                "left":  (0, 0, "persp"),
                "right": (0, 180, "persp"),
                "front": (0, -90, "persp"),
                "back":  (0, 90, "persp"),
            }.items():
                ax.view_init(elev=view[0], azim=view[1])
                ax.set_proj_type(view[2])
                fig.canvas.draw_idle()
                fig.savefig(f"imgs\{name}.png", dpi=200, bbox_inches="tight")
                print(name[0], end=' ')

            print(" ...done!")
            # Create a report after alt-saving
            if config.get("report", {}).get("create_after_save", False): # if report config is present within config data...
                print("Info: ", end="")
                # manually do a python \report_generator
                import report_generator  # this is not best practice.

            # reset the view
            ax.set_proj_type('persp')
            ax.xaxis.pane.set_visible(True)
            ax.yaxis.pane.set_visible(True)
            ax.zaxis.pane.set_visible(True)
            ax.view_init(30, -60)
            fig.canvas.draw_idle()

fig, ax, toolbar = init_plt(config['visual'])
cur = CursorHelper()

ax.set_navigate(False)
fig.canvas.mpl_connect('motion_notify_event', cur._mpl_move)
fig.canvas.mpl_connect('button_press_event', cur._mpl_click)
fig.canvas.mpl_connect('button_release_event', cur._mpl_release)
fig.canvas.mpl_connect('scroll_event', cur._mpl_scroll)
fig.canvas.mpl_connect('key_press_event', mpl_onkey)
container = Container(manifest)
container._render(fig, ax)
plt.show()