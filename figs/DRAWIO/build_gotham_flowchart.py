"""Build the editable GOTHAM algorithm flowchart for diagrams.net.

All visible labels use diagrams.net's MathJax rendering (mxGraphModel math=1).
The terminology and iteration indices follow Algorithm 1 of the current
Operator-Theoretic Homotopy Analysis Method manuscript.
"""

from pathlib import Path
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
OUT = HERE / "GOTHAM_flowchart.drawio"

mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "version": "24.7.17"})
diagram = ET.SubElement(mxfile, "diagram", {"id": "gotham-algorithm", "name": "GOTHAM algorithm"})
model = ET.SubElement(diagram, "mxGraphModel", {
    "dx": "1400", "dy": "1630", "grid": "1", "gridSize": "10",
    "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
    "fold": "1", "page": "1", "pageScale": "1", "pageWidth": "1400",
    "pageHeight": "1630", "math": "1", "shadow": "0",
    "background": "#FFFFFF",
})
root = ET.SubElement(model, "root")
ET.SubElement(root, "mxCell", {"id": "0"})
ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

INK = "#233247"
BLUE = "#2563A6"
PALE = "#EFF6FC"
GRAY = "#64748B"


def node(id_, label, x, y, w, h, kind="process", fill="#FFFFFF", size=18):
    base = (
        f"whiteSpace=wrap;html=1;strokeColor={INK};strokeWidth=2;"
        f"fillColor={fill};fontColor={INK};fontSize={size};"
        "align=center;verticalAlign=middle;spacing=8;"
    )
    shape = {
        "process": "rounded=0;",
        "terminal": "ellipse;perimeter=ellipsePerimeter;",
        "decision": "rhombus;perimeter=rhombusPerimeter;",
        "annotation": f"rounded=0;dashed=1;strokeColor={GRAY};strokeWidth=1.5;",
        "text": "text;strokeColor=none;fillColor=none;",
    }[kind]
    cell = ET.SubElement(root, "mxCell", {
        "id": id_, "value": f"$${label}$$", "style": base + shape,
        "vertex": "1", "parent": "1",
    })
    ET.SubElement(cell, "mxGeometry", {
        "x": str(x), "y": str(y), "width": str(w), "height": str(h),
        "as": "geometry",
    })


def edge(id_, source, target, *, exit=(0.5, 1), entry=(0.5, 0),
         label="", points=(), color=INK, dashed=False):
    style = (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
        f"strokeColor={color};strokeWidth=2;"
        "endArrow=block;endFill=1;html=1;"
        f"exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;"
        f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
    )
    if dashed:
        style += "dashed=1;"
    cell = ET.SubElement(root, "mxCell", {
        "id": id_, "value": f"$${label}$$" if label else "",
        "style": style, "edge": "1", "parent": "1",
        "source": source, "target": target,
    })
    geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    if points:
        arr = ET.SubElement(geom, "Array", {"as": "points"})
        for x, y in points:
            ET.SubElement(arr, "mxPoint", {"x": str(x), "y": str(y)})


node("title", r"\textbf{GOTHAM: iterative Wiener-system identification}",
     210, 20, 980, 50, "text", size=24)
node("start", r"\text{Start}", 600, 95, 200, 60, "terminal", PALE, 20)

node("model", r"""\begin{gathered}\text{Select model parametrization}\\[3pt]
W_{\mathrm{PL}}^{(k)}(\theta)=\widehat\Phi^{(k)}\theta
\quad\text{or}\quad W_{\mathrm{FSM}}(\theta)\end{gathered}""",
     95, 230, 550, 130, fill="#FFFFFF", size=18)
node("loss", r"""\begin{gathered}\text{Select matching loss functional}\\[3pt]
J_{\mathrm{PL}}^{(k)}(\theta)=\tfrac12\|C-\widehat\Phi^{(k)}\theta\|_2^2\\
J_{\nu,\mathrm{FSM}}(\theta)=\tfrac12\sum_t r_{\nu,\mathrm{FSM}}(t;\theta)^2
\end{gathered}""",
     755, 230, 550, 130, fill="#FFFFFF", size=17)

node("root", r"""\begin{gathered}\text{Choose inverse-problem formulation}\\[4pt]
N_{\mathrm G}(\theta)=\nabla_\theta J(\theta)
\quad\text{or}\quad
N_{\mathrm L}(\theta)=J(\theta)-\varepsilon_0
\end{gathered}""",
     330, 450, 740, 120, fill=PALE, size=19)

node("zode", r"""\begin{gathered}\text{Set up the zero-order deformation equation}\\[5pt]
(1-q)\,\mathcal L\!\left(\phi(q)-\theta_0\right)
=q\,\hbar\,\mathcal H\,N\!\left(\phi(q)\right)\\[4pt]
\text{Choose }\mathcal L,\mathcal L^\dagger,\mathcal H,\hbar,\widehat\theta^{(0)},M,K,\delta_{\mathrm{stop}}
\end{gathered}""",
     275, 640, 850, 155, fill="#FFFFFF", size=19)

node("init", r"""\begin{gathered}k\leftarrow0,\quad
\theta_0^{(k)}\leftarrow\widehat\theta^{(k)},\quad m\leftarrow1
\end{gathered}""",
     425, 845, 550, 75, fill=PALE, size=20)

node("outer_test", r"""\begin{gathered}k\ge K\;\text{or}\\
\|N(\widehat\theta^{(k)})\|\le\delta_{\mathrm{stop}}\,?
\end{gathered}""",
     505, 975, 390, 150, "decision", "#FFFDF5", size=18)
node("finish", r"""\begin{gathered}\text{Return}\\
\widehat\theta^{(k)}\end{gathered}""",
     1080, 1020, 230, 75, "terminal", PALE, size=18)

node("inner_test", r"m>M\,?", 510, 1190, 380, 140,
     "decision", "#FFFDF5", size=22)
node("update", r"""\begin{gathered}\widehat\theta^{(k+1)}\leftarrow
\sum_{j=0}^{M}\theta_j^{(k)}\\[4pt]
k\leftarrow k+1,\quad\theta_0^{(k)}\leftarrow\widehat\theta^{(k)},\quad m\leftarrow1
\end{gathered}""",
     55, 1190, 350, 145, fill=PALE, size=17)

node("hode", r"""\begin{gathered}\text{Solve the }m\text{-th HODE}\\[3pt]
R_m^{(k)}=\mathcal D_{m-1}\!\left(N\!\left(\phi^{(k)}\right)\right)\\[3pt]
\theta_m^{(k)}=\chi_m\theta_{m-1}^{(k)}+
\mathcal L^\dagger\!\left(\hbar\mathcal H R_m^{(k)}\right)
\end{gathered}""",
     375, 1400, 650, 145, fill="#FFFFFF", size=18)
node("increment", r"m\leftarrow m+1", 1085, 1435, 225, 65,
     fill=PALE, size=19)

edge("e_start_model", "start", "model", exit=(0.25, 1), entry=(0.5, 0),
     points=((370, 185),))
edge("e_start_loss", "start", "loss", exit=(0.75, 1), entry=(0.5, 0),
     points=((1030, 185),))
edge("e_model_root", "model", "root", exit=(0.5, 1), entry=(0.25, 0),
     points=((370, 405),))
edge("e_loss_root", "loss", "root", exit=(0.5, 1), entry=(0.75, 0),
     points=((1030, 405),))
edge("e_root_zode", "root", "zode")
edge("e_zode_init", "zode", "init")
edge("e_init_outer", "init", "outer_test")
edge("e_outer_finish", "outer_test", "finish", exit=(1, 0.5),
     entry=(0, 0.5), label=r"\text{yes}", color=BLUE)
edge("e_outer_inner", "outer_test", "inner_test", label=r"\text{no}")
edge("e_inner_update", "inner_test", "update", exit=(0, 0.5),
     entry=(1, 0.5), label=r"\text{yes}", color=BLUE)
edge("e_inner_hode", "inner_test", "hode", label=r"\text{no}")
edge("e_hode_increment", "hode", "increment", exit=(1, 0.5),
     entry=(0, 0.5))
edge("e_inner_loop", "increment", "inner_test", exit=(0.5, 0),
     entry=(1, 0.5), points=((1230, 1260),), color=BLUE)
edge("e_outer_loop", "update", "outer_test", exit=(0, 0.5),
     entry=(0, 0.5), points=((25, 1260), (25, 1050)), color=BLUE)

ET.indent(mxfile, space="  ")
ET.ElementTree(mxfile).write(OUT, encoding="utf-8", xml_declaration=True)
print(OUT)
