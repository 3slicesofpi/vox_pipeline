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
        #   
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
            yield "CustomType"
ntype_g = New_Type_Generator()

def dict_grabber(key:str, default:str, *dicts):
    for d in dicts:
        if key in d:
            return d[key]
    print("Warning: Key", key, "not found, default", default, "used.")
    return default

class Container():
        
    def __init__(self, data:dict):  # data is expected to be dict holding all Container data.
        self.Container_ID = uid_e.test("Container_ID",data)
        self.Container_Type = data.get("Container_Type", ntype_g.new())
        td:dict = td_con.get(self.Container_Type, "Default")

        dim:dict = dict_grabber("Dimensions", {}, data, td)
        self.dimLength = dim.get("Length", 0)
        self.dimWidth = dim.get("Width", 0)
        self.dimHeight = dim.get("Height", 0)

        self.Weight = dict_grabber("Weight", 0, data, td)
        self.WeightLimit = dict_grabber("WeightLimit", -1, data, td)

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
            "Weight" : self.Weight,
            "Gross_Weight" : self.Weight+sum(p.Weight for p in self.Packages),
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

    def check_contraints(self):
        for pkg in self.Packages:
            pkg.EdgeColors = Package.d_EdgeColors
        order = [self._check_overlap, self._check_floating, self._check_container_weight, self._check_package_weight, self._check_out_of_bounds]
        text = "Constraint Check:\n\n"

        for check in order:
            outcome, line = check()
            text+=line

        fig, ax = plt.subplots()
        print("Constraints Text Window. This window is safe to close.")
        ax.text(
            0.02, 0.98, 
            text,
            ha='left',
            va='top', 
            family='monospace',
            fontsize=12)
        ax.axis('off')
        plt.show()

    def _check_overlap(self) -> bool|str:
        overlaps = []
        for i, pkg1 in enumerate(self.Packages):
            for pkg2 in self.Packages[i+1:]:
                # Check if bounding boxes overlap in all three dimensions
                if (pkg1.posx < pkg2.posx + pkg2.dimLength and
                    pkg1.posx + pkg1.dimLength > pkg2.posx and
                    pkg1.posy < pkg2.posy + pkg2.dimWidth and
                    pkg1.posy + pkg1.dimWidth > pkg2.posy and
                    pkg1.posz < pkg2.posz + pkg2.dimHeight and
                    pkg1.posz + pkg1.dimHeight > pkg2.posz):
                    overlaps.append(pkg1)
                    overlaps.append(pkg2)
        if overlaps:
            overlapping = set(overlaps)
            for pkg in overlapping:
                pkg.EdgeColors = 'r' 
            return False, f"FAIL: Overlaps: {len(overlaps)}, {len(overlapping)} packages found overlapping.\n"
        else:
            return True, "PASS: No overlaps found.\n"
        
    def _check_floating(self) -> bool|str:
        floaters = []
        for pkg in self.Packages:
            # Check if package is floating (not supported from below)
            # Test 4 corners and midpoint of the bottom face
            test_points = [
                (pkg.posx, pkg.posy),
                (pkg.posx + pkg.dimLength, pkg.posy),
                (pkg.posx + pkg.dimLength, pkg.posy + pkg.dimWidth),
                (pkg.posx, pkg.posy + pkg.dimWidth),
                (pkg.posx + pkg.dimLength/2, pkg.posy + pkg.dimWidth/2)  # midpoint
            ]
            
            is_floating = True
            # Check if package is on the floor or supported by another package
            if pkg.posz == 0:
                is_floating = False
            else:
                for x, y in test_points:
                    for other_pkg in self.Packages:
                        if other_pkg == pkg:
                            continue
                        # Check if this point is supported by another package's top surface
                        if (other_pkg.posx <= x <= other_pkg.posx + other_pkg.dimLength and
                            other_pkg.posy <= y <= other_pkg.posy + other_pkg.dimWidth and
                            other_pkg.posz + other_pkg.dimHeight == pkg.posz):
                            is_floating = False
                            break
                    if not is_floating:
                        break
                    else:
                        pkg.EdgeColors = 'r' 
            
            if is_floating:
                floaters.append(pkg)
                pkg.EdgeColors = 'r'

        if floaters:
            return False, f"FAIL: Floating: {len(floaters)} package(s) not supported.\n"
        else:
            return True, "PASS: No Floating Packages\n"

    def _check_container_weight(self):
        if self.WeightLimit < 1:
            return None, "WARN: Container has no WeightLimit set.\n"
        else:
            weight = sum([pkg.Weight for pkg in self.Packages if pkg.Weight > 1])
        if weight > self.WeightLimit:
            return 0, f"FAIL: Sum Weight of Packages ({weight}) exceeds container limit ({self.WeightLimit})\n"
        else:
            return 1, "PASS: Sum Weight of Packages is: "+weight+"\n"

    def _check_package_weight(self):
        exceeding_packages = []
        for pkg in self.Packages:
            if pkg.WeightLimit < 1:
                continue
            
            test_points = [
                (pkg.posx, pkg.posy),
                (pkg.posx + pkg.dimLength, pkg.posy),
                (pkg.posx + pkg.dimLength, pkg.posy + pkg.dimWidth),
                (pkg.posx, pkg.posy + pkg.dimWidth),
                (pkg.posx + pkg.dimLength/2, pkg.posy + pkg.dimWidth/2)
            ]
            # Test 4 corners and midpoint of the bottom face
            supporting_packages = {}
            for x, y in test_points:
                for other_pkg in self.Packages:
                    if other_pkg == pkg:
                        continue
                    if (other_pkg.posx <= x <= other_pkg.posx + other_pkg.dimLength and
                        other_pkg.posy <= y <= other_pkg.posy + other_pkg.dimWidth and
                        other_pkg.posz + other_pkg.dimHeight == pkg.posz):
                        supporting_packages[other_pkg.Package_ID] = other_pkg
            
            if len(supporting_packages) >= 3:
                total_weight = sum(p.Weight for p in supporting_packages.values())
                if total_weight > pkg.WeightLimit:
                    exceeding_packages.append(pkg)
                    pkg.EdgeColors = 'r'

        if exceeding_packages:
            return False, f"FAIL: Weight Limit Exceeded: {len(exceeding_packages)} package(s) on supporting packages exceeding their limits.\n"
        else:
            return True, "PASS: All package weight limits respected.\n"

    def _check_out_of_bounds(self):
        out_of_bounds = []
        for pkg in self.Packages:
            if (pkg.posx < 0 or pkg.posx + pkg.dimLength > self.dimLength or
                pkg.posy < 0 or pkg.posy + pkg.dimWidth > self.dimWidth or
                pkg.posz < 0 or pkg.posz + pkg.dimHeight > self.dimHeight):
                out_of_bounds.append(pkg)
                pkg.EdgeColors = 'r'

        if out_of_bounds:
            return False, f"FAIL: Out of Bounds: {len(out_of_bounds)} package(s) outside container.\n"
        else:
            return True, "PASS: All packages within container bounds.\n"
        
class Package():
    _C_FACES = CUBE_FACES
    d_Weight = 0.0  # default Weight
    d_FaceColors = ['white'] * 6  # default face colors
    # no facecolor
    f_FaceColors = ['green', 'cyan', 'cyan', 'cyan', 'cyan', 'cyan']   # edge color on focus
    p_FaceColors = ['green', 'cyan', 'cyan', 'cyan', 'cyan', 'cyan'] * 6  # edge color on pick
    d_LineWidth = 1.3  # default line width
    h_LineWidth = 2.5  # line width on hover
    f_LineWidth = 2.5  # line width on focus
    p_LineWidth = 2.5  # line width on pick
    d_EdgeColors = 'k'  # default edge colors
    h_EdgeColors = 'lime'  # edge color on hover
    f_EdgeColors = 'g'  # edge color on focus
    p_EdgeColors = 'k'  # edge color on pick
    d_Alpha = 0.12  # default alpha
    h_Alpha = 0.50  # transparency on hover
    f_Alpha = 0.80  # transparency on focus
    p_Alpha = 0.80  # transparency on pick
    posSnap = config['visual'].get('snap_behaviour', {}).get('pkg_move', 2)
    def __init__(self, data:dict):
        self.Package_ID = uid_e.test("Package_ID",data)
        self.Package_Type = data.get("Package_Type", ntype_g.new())
        td:dict = td_pkg.get(self.Package_Type, "Default")

        dim:dict = dict_grabber("Dimensions", {}, data, td)
        self.dimLength = dim.get("Length", 0)
        self.dimWidth = dim.get("Width", 0)
        self.dimHeight = dim.get("Height", 0)

        self.Weight = dict_grabber("Weight", 0, data, td)
        self.WeightLimit = dict_grabber("WeightLimit", -1, data, td)

        pos:dict = data.get("Position", {})
        self.posx = pos.get("x", 0)
        self.posy = pos.get("y", 0)
        self.posz = pos.get("z", 0)    
        
        color = dict_grabber("Color", self.d_FaceColors, data, td)
        if type(color) == list and len(color) == 6:
            self.FaceColors = color
        elif type(color) == str:
            self.FaceColors = [color] * 6
        else: 
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

    def _update_costume(self, state:str):
        match state:
            case "hover":
                self.polygon.set_edgecolor(self.h_EdgeColors)
                self.polygon.set_facecolor(self.FaceColors)
                self.polygon.set_linewidth(self.h_LineWidth)
                self.polygon.set_alpha(self.h_Alpha)
            case "focus":
                self.polygon.set_edgecolor(self.f_EdgeColors)
                self.polygon.set_facecolor(self.f_FaceColors)
                self.polygon.set_linewidth(self.f_LineWidth)
                self.polygon.set_alpha(self.f_Alpha)
            case "pick":
                self.polygon.set_edgecolor(self.p_EdgeColors)
                self.polygon.set_facecolor(self.p_FaceColors)
                self.polygon.set_linewidth(self.p_LineWidth)
                self.polygon.set_alpha(self.p_Alpha)
            case _:
                self._reset_costume()
    
    def _reset_costume(self):
        self.polygon.set_edgecolor(self.EdgeColors)
        self.polygon.set_facecolor(self.FaceColors)
        self.polygon.set_linewidth(self.LineWidth)
        self.polygon.set_alpha(self.Alpha)

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
    
    def _moveTo(self, posx:float=None, posy:float=None, posz:float=None):
        if posx is not None and 0 < posx <= container.dimLength-self.dimLength : self.posx = round(posx, self.posSnap) 
        if posy is not None and 0 < posy <= container.dimWidth-self.dimWidth : self.posy = round(posy, self.posSnap) 
        if posz is not None and 0 < posz <= container.dimHeight-self.dimHeight : self.posz = round(posz, self.posSnap)
        self.update_pos()

    def _render(self, fig, ax):
        ax.add_collection3d(self.polygon)
        self._update_costume("default")

    def _export(self):
        return {
            "Package_ID" : self.Package_ID,
            "Package_Type": self.Package_Type,
            "Dimensions" : {"Length" : self.dimLength, "Width" : self.dimWidth, "Height" : self.dimHeight},
            "Weight" : self.Weight,
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
            color="black", linewidth=1.3
            ) 

    def rotate_pkg(self):
        if self.focuspkg:
            length = self.focuspkg.dimLength
            width = self.focuspkg.dimWidth
            self.focuspkg.dimLength = width
            self.focuspkg.dimWidth = length
            self.focuspkg.update_pos()
    
    def pitch_pkg(self):
        if self.focuspkg:
            height = self.focuspkg.dimHeight
            width = self.focuspkg.dimWidth
            self.focuspkg.dimHeight = width
            self.focuspkg.dimWidth = height
            self.focuspkg.update_pos()
    
    def yaw_pkg(self):
        if self.focuspkg:
            length = self.focuspkg.dimLength
            height = self.focuspkg.dimHeight
            self.focuspkg.dimLength = height
            self.focuspkg.dimHeight = length
            self.focuspkg.update_pos()

    def _update(self):
        for pkg in container.Packages:
            pkg._update_costume("default")
        for pkg in self.hoverpkg:
            pkg._update_costume("hover")
        if self.focuspkg:
            self.focuspkg._update_costume("focus")
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
                self.focuspkg._moveTo(posx=self.posx, posy=self.posy)
                self._update()

    def _mpl_click(self, event):
        if not event.inaxes:
            return
        elif self.focuspkg is not None and self.state == "hover" and event.button==3:
            self.focuspkg._update_costume("pick")
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
                    self.posz += 0.1; self.focuspkg._moveTo(posz=self.posz)
                    self.focuspkg.update_pos()
                    self._update(); return
                elif event.button == 'down' and self.posz - 0.1 >= 0:
                    self.posz -= 0.1; self.focuspkg._moveTo(posz=self.posz)
                    self.focuspkg.update_pos()
                    self._update(); return

def save_container():
    print("Info: Saving new manifest...")
    container.export_tofile()
    print("Info: Saving Load Views... ", end='')
    # prepare a new view
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    cur.probeline.set_visible(False)
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
    cur.probeline.set_visible(True)
    ax.view_init(30, -60)
    fig.canvas.draw_idle()

def toggle_package_alpha():
    alpha = 0
    print("Changed Package Transparency. Move cursor to take effect.")
    while True:
        match alpha:
            case 0:
                alpha = 1
            case 1:
                alpha = Package.d_Alpha
            case Package.d_Alpha:
                alpha = 0
        for p in container.Packages:
            p.Alpha = alpha
            p.polygon.set_alpha(alpha)
        fig.canvas.draw_idle()
        yield None

Obj_toggle_package_alpha = toggle_package_alpha()
def tog_pkg_alpha():
    return next(Obj_toggle_package_alpha)


def mpl_onkey(event):
    match event.key:
        case 'ctrl+s':
            print("Info: Saved Current VIEW")
            print("Info: To save edited manifest, press [ALT]+[S].")
        case 'alt+s':
            save_container()
        case 'alt+t':
            tog_pkg_alpha()
        case 'r':
            cur.rotate_pkg()
        case 't':
            cur.pitch_pkg()
        case 'y':
            cur.yaw_pkg()
        case 'c':
            container.check_contraints()

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