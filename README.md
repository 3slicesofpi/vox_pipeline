
## Overview
(part of) 3D Package Loader System
Visualiser, Dashboard/Report Generator

## Controls

Move Mouse - Hover(Yellow Borders) and Select(Highlighted in Cyan) 
R-Click - Drag Selected Package 
R-Click + Scroll - Drag Selected Package up and down.
S - Save View
Alt-S - Save Load and View Report
C - Check Load Constraints

R - Rotate
T - Pitch
Y - Yaw


## Features

- **3D Visualiser**: Interactive 3D visualization tools
- **Dashboard/Report Generator**: Create and generate comprehensive reports: html, pdf
- **Free to Use**: Free to download and use. 
- **Drag and Drop**: Users can freely move(right-click), rotate(R), pitch(T) and yaw(Y) their packages to customize load.
- **Viewing and Interaction modes**: ✥ Button used to toggle between moving (pan, limited zoom) and interaction (moving packages)
- **Reset View**: ⌂ Button used to reset view after panning or zooming. 
- **Constraints Check**: Users can check if current Package Load meets constraints with C. Program will display warnings if not met. 
- **Save View**: Users can save the current view with S
- **Save and Display Load**: Users can save current loading with Alt-S. A Loading Report will also be generated. Save that with Ctrl-P+Enter.

## Limitations

- **2.5D User Interaction**: Movement of the cursor is limited to a 2d "floor" plane. Use scroll wheel to move the cursor up and down.
- **Render Speed**: Above 100 packages, lag may occur. 
- **Data Assumptions**: The program assumes all user-inputted data is correct.  
- **One Container**: The program supports displaying only one container. 

## Getting Started

Ensure all dependencies are set up: 
pip install -r requirements.txt

Run:
python ./exec.py
python3 ./exec.py

Note: Due to technical limitations, it is not reccomended to run on Google Colab.

## Technical Information

Visualiser's GUI runs on a single matplotlib figure window. 

The Axis Limits are configured to reflect the Containers own internal dimensions. Packages exist within these limits.

The buttons found on the figure window have been repurposed. 
Home ⌂: Reset View (useful to return to original view after edits)
Move ✥: Toggles between Viewing (pan and zoom) and Interaction (right click and scroll) modes. 
Search 🔎︎: Disabled (too disorienting)
Users may manually save current view with the save button. 

The Report Generator procedually generates a new HTML file for the saved Load. The static HTML file is served to users. Users can save as PDF or print using Ctrl-P. 

It is theoretically possible to run on Jupyter. Try with the Qt backend (currently Tk)
It is not feasible to run on Google Colab. Colab does not support GUI windows at all. 

Program assumes all user-inputted data is correct, and will "warn" or crash rather than "fix" problems.

Program uses Fallback Resolution. Parser behaviour is as follows:
Input file contains multiple PackageData instances. If searched value is found, use it, if not, check if PackageType parameter is available. If it is, use PackageTypeData.json and view its value. If both are not available, use a default value. This allows for fault tolerance and simpler Input file structures (values can be stored in PackageType, not in input file.)

## Contributing

Contributions are welcome. Please submit issues and pull requests to the repository.
