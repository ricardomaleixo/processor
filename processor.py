# processor.py
import os
import re
import math

import numpy as np

from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.GProp import GProp_GProps
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCC.Core.gp import gp_Vec
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location


# ----------------------------------------------------------------------
# 1️⃣ Leitura de STEP (shape + metadados)
# ----------------------------------------------------------------------
def _parse_metadata_from_file_text(file_path: str) -> dict:
    """
    Lê o HEADER do arquivo STEP como texto e tenta extrair:
    - Part (a partir de FILE_NAME ou do nome do arquivo)
    - Material, Quantity, Status, Status reason (a partir da FILE_DESCRIPTION)
      usando o padrão: 'Material=..., Quantity=..., Status=..., StatusReason=...'
    """

    metadata = {
        "Part": os.path.basename(file_path).split(".")[0],
        "Material": "",
        "Quantity": 1,
        "Status": "Finished",
        "Status reason": "",
        "Schema": "",
        "Raw description": "",
    }

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return metadata

    # HEADER ... ENDSEC;
    m_header = re.search(
        r"HEADER\s*;(.*?);ENDSEC\s*;",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    header = m_header.group(1) if m_header else content

    # FILE_NAME('Nome', ...)
    m_name = re.search(r"FILE_NAME\s*\(\s*'([^']*)'", header, re.IGNORECASE)
    if m_name:
        name = m_name.group(1).strip()
        if name:
            metadata["Part"] = name

    # FILE_DESCRIPTION(('Texto'),...)
    m_desc = re.search(
        r"FILE_DESCRIPTION\s*\(\s*\(\s*'([^']*)'\s*\)",
        header,
        re.IGNORECASE | re.DOTALL,
    )
    desc = m_desc.group(1).strip() if m_desc else ""
    metadata["Raw description"] = desc

    # Exemplo: "Material=INOX 304; Quantity=2; Status=Finished; StatusReason=..."
    matches = re.findall(r"(\w+)\s*=\s*([^;]+)", desc)
    for k, v in matches:
        k = k.strip().lower()
        v = v.strip()
        if k == "material":
            metadata["Material"] = v
        elif k == "quantity":
            try:
                metadata["Quantity"] = int(float(v))
            except Exception:
                pass
        elif k == "status":
            metadata["Status"] = v
        elif k in ("statusreason", "status_reason"):
            metadata["Status reason"] = v

    # FILE_SCHEMA(('schema'))
    m_schema = re.search(
        r"FILE_SCHEMA\s*\(\s*\(\s*'([^']*)'\s*\)\s*\)",
        header,
        re.IGNORECASE | re.DOTALL,
    )
    if m_schema:
        metadata["Schema"] = m_schema.group(1).strip()

    return metadata


def read_step_file(file_path: str):
    """
    Lê o arquivo STEP uma única vez e retorna:
    - shape principal (via pythonOCC)
    - metadados extraídos do texto do arquivo (HEADER)
    """
    reader = STEPControl_Reader()
    status = reader.ReadFile(file_path)
    if status != IFSelect_RetDone:
        raise ValueError(f"Erro ao ler o arquivo STEP: {file_path}")

    reader.TransferRoots()
    shape = reader.OneShape()

    metadata = _parse_metadata_from_file_text(file_path)
    return shape, metadata


# ----------------------------------------------------------------------
# 2️⃣ Funções geométricas básicas
# ----------------------------------------------------------------------
def edge_length(edge) -> float:
    """
    Calcula o comprimento de uma aresta usando GCPnts_AbscissaPoint.
    """
    curve, first, last = BRep_Tool.Curve(edge)
    if curve is None:
        return 0.0

    adaptor = GeomAdaptor_Curve(curve, first, last)
    try:
        length = GCPnts_AbscissaPoint.Length(adaptor, first, last)
        return float(length)
    except Exception:
        return 0.0


def face_is_plane(face) -> bool:
    """
    Verifica se uma face é plana.
    """
    surf = BRepAdaptor_Surface(face)
    return surf.GetType() == GeomAbs_Plane


def angle_between_faces(f1, f2) -> float:
    """
    Retorna o ângulo entre duas faces (em graus), no intervalo [0, 180].
    """
    surf1 = BRepAdaptor_Surface(f1)
    surf2 = BRepAdaptor_Surface(f2)

    n1 = surf1.Plane().Axis().Direction()
    n2 = surf2.Plane().Axis().Direction()

    dot = n1.Dot(n2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def estimate_thickness(
    faces,
    default_thickness: float = 3.0,
    min_thickness: float = 0.1,
    max_thickness: float = 100.0,
) -> float:
    """
    Estima a espessura como a menor distância entre pares de faces planas
    aproximadamente paralelas. Se não conseguir, retorna default_thickness.
    """
    if not faces:
        return default_thickness

    min_distance = None
    n_faces = len(faces)

    for i in range(n_faces):
        surf1 = BRepAdaptor_Surface(faces[i])
        if surf1.GetType() != GeomAbs_Plane:
            continue
        pln1 = surf1.Plane()
        n1 = pln1.Axis().Direction()
        p1 = pln1.Location()

        for j in range(i + 1, n_faces):
            surf2 = BRepAdaptor_Surface(faces[j])
            if surf2.GetType() != GeomAbs_Plane:
                continue
            pln2 = surf2.Plane()
            n2 = pln2.Axis().Direction()
            p2 = pln2.Location()

            dot = abs(n1.Dot(n2))
            if dot > math.cos(math.radians(1)):  # ~1 grau
                vec_p1p2 = gp_Vec(p1, p2)
                n1_vec = gp_Vec(n1.XYZ())
                d = abs(vec_p1p2.Dot(n1_vec))
                if min_thickness < d < max_thickness:
                    if min_distance is None or d < min_distance:
                        min_distance = d

    return round(min_distance, 3) if min_distance is not None else default_thickness


def calculate_cutting_length(shape, planar_faces) -> float:
    """
    Soma o comprimento das arestas externas das faces planas (aresta com apenas 1 face).
    """
    edge_to_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces)

    cutting_edges = set()
    for f in planar_faces:
        expE = TopExp_Explorer(f, TopAbs_EDGE)
        while expE.More():
            e = expE.Current()
            if edge_to_faces.Contains(e):
                faces = edge_to_faces.FindFromKey(e)
                if faces.Size() == 1:  # aresta externa
                    cutting_edges.add(e)
            expE.Next()

    total_length = sum(edge_length(e) for e in cutting_edges)
    return round(total_length, 2)


def get_material_density(material: str, default_density: float = 7.9e-6) -> float:
    """
    Retorna uma densidade aproximada em kg/mm³ baseada na string de material.
    """
    if not material:
        return default_density

    m = material.lower()
    if "inox" in m or "steel" in m or "aço" in m:
        return 7.9e-6
    if "alum" in m:
        return 2.7e-6
    if "cobre" in m or "copper" in m:
        return 8.9e-6

    return default_density


def _count_bend_sequences(bend_edges):
    """
    Agrupa arestas de dobra em sequências contínuas (componentes conexos).
    """
    if not bend_edges:
        return 0

    vertex_to_edges = {}
    for idx, e in enumerate(bend_edges):
        expV = TopExp_Explorer(e, TopAbs_VERTEX)
        while expV.More():
            v = expV.Current()
            vertex_to_edges.setdefault(v, set()).add(idx)
            expV.Next()

    adj = {i: set() for i in range(len(bend_edges))}
    for edges_at_v in vertex_to_edges.values():
        edges_list = list(edges_at_v)
        for i in range(len(edges_list)):
            for j in range(i + 1, len(edges_list)):
                a = edges_list[i]
                b = edges_list[j]
                adj[a].add(b)
                adj[b].add(a)

    visited = [False] * len(bend_edges)
    num_sequences = 0

    for i in range(len(bend_edges)):
        if not visited[i]:
            num_sequences += 1
            stack = [i]
            visited[i] = True
            while stack:
                cur = stack.pop()
                for nb in adj[cur]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)

    return num_sequences


# ----------------------------------------------------------------------
# 2.1 – Bounding box via PCA
# ----------------------------------------------------------------------
def get_bbox_pca(shape, linear_deflection: float = 0.1, angular_deflection: float = 0.1):
    """
    Bounding box ORIENTADO (OBB) usando PCA nos pontos da malha.
    Retorna (Lmax, Lmid, Lmin).
    """
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesh.Perform()

    points = []

    expF = TopExp_Explorer(shape, TopAbs_FACE)
    while expF.More():
        face = expF.Current()
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation is not None:
            nb_nodes = triangulation.NbNodes()
            trsf = loc.Transformation()
            for i in range(1, nb_nodes + 1):
                p = triangulation.Node(i)
                if not loc.IsIdentity():
                    p = p.Transformed(trsf)
                points.append([p.X(), p.Y(), p.Z()])
        expF.Next()

    # Fallback: bbox axis-aligned
    if not points:
        box = Bnd_Box()
        brepbndlib.Add(shape, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        dims = sorted([dx, dy, dz], reverse=True)
        return float(dims[0]), float(dims[1]), float(dims[2])

    pts = np.asarray(points, dtype=float)

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    coords = centered @ eigvecs
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    lengths = maxs - mins

    lengths_sorted = sorted(lengths, reverse=True)
    Lmax, Lmid, Lmin = lengths_sorted
    return float(Lmax), float(Lmid), float(Lmin)


# ----------------------------------------------------------------------
# 2.2 – Bounding box orientado pela MAIOR face planar
# ----------------------------------------------------------------------
def get_bbox_main_face(shape, linear_deflection: float = 0.1, angular_deflection: float = 0.1):
    """
    Bounding box orientado pela MAIOR face planar.
    Retorna (width_2d, height_2d, depth_3d) no sistema de eixos dessa face.
    """
    # 1) Encontrar maior face plana
    main_face = None
    max_area = 0.0

    expF = TopExp_Explorer(shape, TopAbs_FACE)
    while expF.More():
        f = expF.Current()
        if face_is_plane(f):
            gprops = GProp_GProps()
            brepgprop.SurfaceProperties(f, gprops)
            area = gprops.Mass()
            if area > max_area:
                max_area = area
                main_face = f
        expF.Next()

    # Se não achou face plana, cai para PCA
    if main_face is None:
        Lmax, Lmid, Lmin = get_bbox_pca(shape, linear_deflection, angular_deflection)
        # aqui devolvemos 2D = duas maiores dimensões, depth = menor
        return float(Lmid), float(Lmax), float(Lmin)

    # 2) Sistema de eixos da face
    surf = BRepAdaptor_Surface(main_face)
    pln = surf.Plane()

    origin = pln.Location()
    x_axis = gp_Vec(pln.XAxis().Direction().XYZ())
    y_axis = gp_Vec(pln.YAxis().Direction().XYZ())
    z_axis = gp_Vec(pln.Axis().Direction().XYZ())  # normal

    # 3) Gera malha
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesh.Perform()

    # 4) Projeta todos os nós da malha
    first = True
    x_min = y_min = z_min = None
    x_max = y_max = z_max = None

    expF = TopExp_Explorer(shape, TopAbs_FACE)
    while expF.More():
        face = expF.Current()
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i)
                if not loc.IsIdentity():
                    p = p.Transformed(trsf)

                v = gp_Vec(origin, p)
                xx = v.Dot(x_axis)
                yy = v.Dot(y_axis)
                zz = v.Dot(z_axis)

                if first:
                    x_min = x_max = xx
                    y_min = y_max = yy
                    z_min = z_max = zz
                    first = False
                else:
                    x_min = min(x_min, xx)
                    x_max = max(x_max, xx)
                    y_min = min(y_min, yy)
                    y_max = max(y_max, yy)
                    z_min = min(z_min, zz)
                    z_max = max(z_max, zz)
        expF.Next()

    if first:
        # se por algum motivo não coletamos pontos, fallback para PCA
        Lmax, Lmid, Lmin = get_bbox_pca(shape, linear_deflection, angular_deflection)
        return float(Lmid), float(Lmax), float(Lmin)

    length_x = abs(x_max - x_min)
    length_y = abs(y_max - y_min)
    length_z = abs(z_max - z_min)

    in_plane_1 = length_x
    in_plane_2 = length_y
    depth = length_z

    # width_2d = menor lado no plano, height_2d = maior lado no plano
    w_2d, h_2d = sorted([in_plane_1, in_plane_2])
    return float(w_2d), float(h_2d), float(depth)


# ----------------------------------------------------------------------
# 2.3 – Wrapper: escolhe PCA ou face
# ----------------------------------------------------------------------
def get_oriented_bbox(shape, mode: str = "pca"):
    """
    Retorna um dict com:
      - width
      - height
      - depth
      - width_2d
      - height_2d

    mode="pca"  → usa PCA (OBB genérico)
    mode="face" → usa a maior face planar como referência
    """
    mode = (mode or "pca").lower()

    if mode == "face":
        w2d, h2d, d = get_bbox_main_face(shape)
        # três dimensões 3D reais
        d1, d2, d3 = w2d, h2d, d
    else:
        # PCA
        Lmax, Lmid, Lmin = get_bbox_pca(shape)
        # aqui consideramos como se as duas maiores fossem "plano principal"
        w2d, h2d = Lmid, Lmax
        d1, d2, d3 = Lmax, Lmid, Lmin

    dims3 = sorted([d1, d2, d3], reverse=True)  # [Lmax, Lmid, Lmin]
    bbox_height = dims3[0]  # maior
    bbox_width = dims3[1]   # intermediária
    bbox_depth = dims3[2]   # menor

    return {
        "width": bbox_width,
        "height": bbox_height,
        "depth": bbox_depth,
        "width_2d": w2d,
        "height_2d": h2d,
    }


# ----------------------------------------------------------------------
# 3️⃣ Extração completa de propriedades da peça (chapas + sólidos)
# ----------------------------------------------------------------------
def get_shape_properties(
    file_path: str,
    default_thickness: float = 3.0,
    bbox_mode: str = "pca",  # "pca" ou "face"
) -> dict:
    """
    Lê a peça de um arquivo STEP e calcula um conjunto de propriedades
    geométricas e tecnológicas.

    bbox_mode:
      - "pca"  → bounding box via PCA (genérico)
      - "face" → bounding box orientado pela maior face planar
    """
    shape, metadata = read_step_file(file_path)

    # Volume (mm³) e área de superfície (mm²)
    vol_props = GProp_GProps()
    brepgprop.VolumeProperties(shape, vol_props)
    volume = vol_props.Mass()

    surf_props = GProp_GProps()
    brepgprop.SurfaceProperties(shape, surf_props)
    surface_area = surf_props.Mass()

    # Bounding box orientado conforme modo
    bbox_info = get_oriented_bbox(shape, mode=bbox_mode)
    bbox_width = bbox_info["width"]
    bbox_height = bbox_info["height"]
    bbox_depth = bbox_info["depth"]
    bb_2d_width = bbox_info["width_2d"]
    bb_2d_height = bbox_info["height_2d"]

    # Faces planas
    expF = TopExp_Explorer(shape, TopAbs_FACE)
    planar_faces = []
    while expF.More():
        f = expF.Current()
        if face_is_plane(f):
            planar_faces.append(f)
        expF.Next()

    thickness = estimate_thickness(
        planar_faces,
        default_thickness=default_thickness,
        min_thickness=0.1,
        max_thickness=100.0,
    )

    num_contours = len(planar_faces)

    # Mapa aresta → faces
    edge_to_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces)

    # Dobras
    bend_edges = []
    bend_lengths = []

    expE = TopExp_Explorer(shape, TopAbs_EDGE)
    while expE.More():
        e = expE.Current()
        if edge_to_faces.Contains(e):
            faces = edge_to_faces.FindFromKey(e)
            if faces.Size() == 2:
                f1 = faces.First()
                f2 = faces.Last()
                if face_is_plane(f1) and face_is_plane(f2):
                    ang = angle_between_faces(f1, f2)
                    if 5.0 < ang < 175.0:
                        L = edge_length(e)
                        if L > 0:
                            bend_edges.append(e)
                            bend_lengths.append(L)
        expE.Next()

    num_bends = len(bend_lengths)
    num_bend_sequences = _count_bend_sequences(bend_edges)
    length_longest_bend = max(bend_lengths) if bend_lengths else 0.0
    force_longest_bend = round(length_longest_bend * thickness * 0.15, 2)

    cutting_length = calculate_cutting_length(shape, planar_faces)

    density = get_material_density(metadata.get("Material", ""))
    mass_kg = volume * density

    bruto_area = bb_2d_width * bb_2d_height

    data = {
        "Status": metadata.get("Status", "Finished"),
        "Status reason": metadata.get("Status reason", ""),
        "Quantity": metadata.get("Quantity", 1),
        "Part": metadata.get("Part", os.path.basename(file_path).split(".")[0]),
        "Material": metadata.get("Material", ""),
        "Thickness": thickness,

        # 2D (no plano principal escolhido pelo modo)
        "Min Bounding box Width": round(bb_2d_width, 2),
        "Min Bounding box Height": round(bb_2d_height, 2),

        # 3D (ordenado: Height = maior, Width = intermediária, Depth = menor)
        "Bounding box 3D Width": round(bbox_width, 2),
        "Bounding box 3D Height": round(bbox_height, 2),
        "Bounding box 3D Depth": round(bbox_depth, 2),

        "Cutting length": cutting_length,
        "Number of contours": num_contours,
        "Number of bends": num_bends,
        "Number of bend sequences": num_bend_sequences,
        "Length longest bend": round(length_longest_bend, 2),
        "Force longest bend": force_longest_bend,

        "Netto area": round(surface_area, 3),
        "Bruto area": round(bruto_area, 3),
        "Mass": round(mass_kg, 3),
        "Cutting": "00:00:00",

        "Schema": metadata.get("Schema", ""),
        "Description": metadata.get("Raw description", ""),
        "Volume": round(volume, 3),
    }

    return data
