import numpy as np
from typing import NamedTuple, Callable, Sequence
import svgwrite
import pyrr


class Viewport(NamedTuple):
    minx: float = -0.5
    miny: float = -0.5
    width: float = 1.0
    height: float = 1.0


class Camera(NamedTuple):
    view: np.ndarray
    projection: np.ndarray


class Mesh(NamedTuple):
    faces: np.ndarray
    style: dict = None
    shader: Callable[[int, float], dict] = None


class Scene(NamedTuple):
    meshes: Sequence[Mesh]


class Engine:
    def __init__(self, views):
        self.views = views

    def render(self, filename, size=(512, 512), viewBox='-0.5 -0.5 1.0 1.0'):
        drawing = svgwrite.Drawing(filename, size, viewBox=viewBox)
        for view in self.views:
            projection = np.dot(view.camera.view, view.camera.projection)
            for mesh in view.scene.meshes:
                drawing.add(self._create_group(drawing, projection, view.viewport, mesh))
        drawing.save()

    def _create_group(self, drawing, projection, viewport, mesh):
        faces = mesh.faces
        shader = mesh.shader or (lambda face_index, winding: {})
        default_style = mesh.style or {}

        ones = np.ones(faces.shape[:2] + (1,))
        faces = np.dstack([faces, ones])
        faces = np.dot(faces, projection)

        faces[:, :, :3] /= faces[:, :, 3:4]
        faces = faces[:, :, :3]

        faces[:, :, 0:1] = (1.0 + faces[:, :, 0:1]) * viewport.width / 2
        faces[:, :, 1:2] = (1.0 + faces[:, :, 1:2]) * viewport.height / 2
        faces[:, :, 0:1] += viewport.minx
        faces[:, :, 1:2] += viewport.miny

        z_centroids = -np.sum(faces[:, :, 2], axis=1)
        for face_index in range(len(z_centroids)):
            z_centroids[face_index] /= len(faces[face_index])
        face_indices = np.argsort(z_centroids)
        faces = faces[face_indices]

        group = drawing.g(**default_style)
        face_index = 0
        for face in faces:
            p0, p1, p2 = face[0], face[1], face[2]
            winding = pyrr.vector3.cross(p2 - p0, p1 - p0)[2]
            style = shader(face_indices[face_index], winding)
            if style != None:
                group.add(drawing.polygon(face[:, 0:2], **style))
            face_index = face_index + 1

        return group


def octahedron():
    """Construct an eight-sided polyhedron"""
    f = np.sqrt(2.0) / 2.0
    verts = np.float32([(0, -1, 0), (-f, 0, f), (f, 0, f), (f, 0, -f), (-f, 0, -f), (0, 1, 0)])
    triangles = np.int32([(0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 1, 4), (5, 1, 2), (5, 2, 3), (5, 3, 4), (5, 4, 1)])
    return verts[triangles]


projecction_matrix = pyrr.matrix44.create_perspective_projection(fovy=25, aspect=1, near=10, far=100)
view_matrix = pyrr.matrix44.create_look_at(eye=[25, -20, 60], target=[0, 0, 0], up=[0, 1, 0])
camera = Camera(view_matrix, projecction_matrix)
style = dict(fill='white', fill_opacity='0.75', stroke='black', stroke_linejoin='round', stroke_width='0.005')
mesh = Mesh(15.0 * octahedron(), style=style)
view = Viewport(camera, Scene[mesh])
Engine([view]).render(('octahedron.svg'))
