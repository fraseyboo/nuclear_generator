import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


import vtk
import LUT_utils

def write_gltf(source, savename='nuclear_shape.gltf', verbose=True):

    if verbose:
        print('writing GLTF to %s' % savename)
    try:
        exporter = vtk.vtkGLTFExporter()
    except AttributeError:
        print('Gltf exporting is not supported in your version of VTK, try updating')
    exporter.SetInput(source)
    exporter.InlineDataOn()
    exporter.SetFileName(savename)
    exporter.Update()
    exporter.Write()
    if verbose:
        print('file written')

def render(actors=None, background_color='White',window_size=(600, 600)):

    renderer = vtk.vtkRenderer()

    render_camera = renderer.GetActiveCamera()

    renderWindow = vtk.vtkRenderWindow()

    renderWindow.SetSize(window_size)
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    # renderWindowInteractor.SetInteractorStyle(MyInteractorStyle(renderWindowInteractor, render_camera, renderWindow))
    renderer.SetBackground(vtk.vtkNamedColors().GetColor3d(background_color))

    renderer.SetUseDepthPeeling(1)
    renderer.SetMaximumNumberOfPeels(10)


    if actors is not None:
        if isinstance(actors, list):
            print('List of actors supplied, adding list')
            for actor in actors:
                renderer.AddActor(actor)
        elif isinstance(actors, dict):
            print('List of actors supplied, adding list')
            for actor in actors.values():
                if isinstance(actor, list):
                    for sub_actor in actor:
                        renderer.AddActor(sub_actor)
                elif type(actor) is list:
                    for sub_actor in actor:
                        renderer.AddActor(sub_actor)
                elif isinstance(actor, dict):
                    for sub_actor in actor.values():
                        renderer.AddActor(sub_actor)
                elif isinstance(actor, tuple):
                    for sub_actor in actor:
                        renderer.AddActor(sub_actor)
                else:
                    renderer.AddActor(actor)

    else:
        renderer.AddActor(actors)


    renderer.ResetCamera()
    renderWindow.Render()
    renderWindowInteractor.Start()

    return renderer, renderWindow, renderWindowInteractor

def translate_point(location, translation=[0,0,0], rotation=[0,0,0]):

    if rotation is not None:
        r_matrix = R.from_euler('xyz', rotation)
        new_loc_1 = r_matrix.apply(location)
    else:
        new_loc_1 = location

    if translation is not None:
        new_loc_2 = new_loc_1 + translation
    else:
        new_loc_2 = new_loc_1
    return new_loc_2


def add_polyhedron(vertices, faces, labels=None, offset=[0, 0, 0], scalars=None, secondary_offset=None, rotation=None, opacity=1.0, verbose=False, mesh_color='black', color_map='viridis', c_range=None, representation='surface', interpolate_scalars=True):

    colors = vtk.vtkNamedColors()

    points = vtk.vtkPoints()
    visited = list()
    narrowed_vertices = list()

    if labels is not None:
        if verbose:
            print('assuming points need relabelling reduction')
        for face in faces:
            for input_vertex in face:
                if input_vertex not in visited:
                    vertex = vertices[np.argwhere(labels == input_vertex), :].flatten() + offset
                    narrowed_vertices.append(vertex)
                    visited.append(input_vertex)

        narrowed_vertices = np.asarray(narrowed_vertices)
        visited = np.asarray(visited)
        new_faces = list()

        for face in faces:
            new_face = [np.argwhere(visited == face[0]), np.argwhere(visited == face[1]), np.argwhere(visited == face[2])]
            new_faces.append(new_face)

        faces = np.squeeze(np.asarray(new_faces))
    else:
        narrowed_vertices = np.asarray(vertices) + offset
    for vertex in narrowed_vertices:
        vertex = translate_point(vertex, secondary_offset, rotation)
        points.InsertNextPoint(vertex[0], vertex[1], vertex[2])

    cell_array = vtk.vtkCellArray()

    if faces.shape[1] == 3:
        for face in faces:
            Triangle = vtk.vtkTriangle()
            Triangle.GetPointIds().SetId(0,face[0])
            Triangle.GetPointIds().SetId(1,face[1])
            Triangle.GetPointIds().SetId(2,face[2])
            cell_array.InsertNextCell(Triangle)

    elif faces.shape[1] == 4:
        for face in faces:
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0,face[0])
            quad.GetPointIds().SetId(1,face[1])
            quad.GetPointIds().SetId(2,face[2])
            quad.GetPointIds().SetId(3,face[3])
            cell_array.InsertNextCell(quad)

    else:
        for face in faces:
            cell = vtk.vtkPolygon()
            for p_num, point in enumerate(face):
                cell.GetPointIds().SetId(p_num, point)
            cell_array.InsertNextCell(cell)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cell_array)

    # normal_filter = vtk.vtkPolyDataNormals()
    # normal_filter.SetInputData(polydata)
    # normal_filter.Update()

    # print(normal_filter.GetOutput(), scalars.shape)
    # Create a mapper and actor
    mapper = vtk.vtkDataSetMapper()
    # mapper.SetInputData(normal_filter.GetOutput())
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d(mesh_color))

    if scalars is not None:

        if c_range is None:
            c_range = (np.min(scalars), np.max(scalars))
            
        lut = LUT_utils.make_LUT(colormap=color_map, c_range=c_range, nan_color=(0,1,0,1))

        actor.GetMapper().SetLookupTable(lut)
        actor.GetMapper().SetScalarRange(c_range)
        
        cur_color_data = vtk.vtkUnsignedCharArray()
        cur_color_data.SetNumberOfComponents(3)
        cur_color_data.SetName("Colors")
        for val in scalars:
            col = [0., 0., 0.]
            lut.GetColor(val, col)
            col = [int(c * 255) for c in col]
            cur_color_data.InsertNextTuple3(col[0], col[1], col[2])

        actor.GetMapper().GetInput().GetPointData().SetScalars(cur_color_data)
        actor.GetMapper().SetInterpolateScalarsBeforeMapping(interpolate_scalars)

    actor.GetProperty().SetOpacity(opacity)

    if representation == 'wireframe':

        actor.GetProperty().SetRepresentationToWireframe()

    return actor


def add_2D_function(function_values, x_range=[0,1], y_range=[0,1], scale=1, scale_mesh=True, add_gridlines=True, mesh_color='grey', absolute_displacement=False, offset=[0,0,0], line_width=2, verbose=False):

    if verbose:
        print(function_values.shape)

    x_granularity = function_values.shape[0]
    y_granularity = function_values.shape[1]

    x_steps, y_steps = np.mgrid[x_range[0]:x_range[1]:(x_granularity)*1j, y_range[0]:y_range[1]:(y_granularity)*1j]

    coordination = np.reshape(np.arange(0,(x_granularity*y_granularity)), [x_granularity, y_granularity])

    if verbose:
        print(coordination)

    triangles = list()

    if add_gridlines:
        quads = list()

    for x_num in range(x_granularity-1):
        for y_num in range(y_granularity-1):

            tri_1 = [coordination[x_num, y_num], coordination[x_num + 1, y_num], coordination[x_num + 1, y_num + 1]]
            triangles.append(tri_1)
            tri_2 = [coordination[x_num, y_num], coordination[x_num + 1, y_num + 1], coordination[x_num , y_num +1]]
            triangles.append(tri_2)

            if add_gridlines:
                quad_1 = [coordination[x_num, y_num], coordination[x_num + 1, y_num], coordination[x_num + 1, y_num + 1], coordination[x_num, y_num + 1]]
                quads.append(quad_1)

    triangles = np.asarray(triangles)

    if add_gridlines:
        quads = np.asarray(quads)

    x = x_steps.flatten()
    y = y_steps.flatten()
    z = function_values.flatten()

    locs = np.asarray([x, y, z]).T
    surface = add_polyhedron(locs, triangles, scalars=z, opacity=1.0, color_map='jet', offset=offset)
    actor_dict = dict()
    actor_dict.update({'Surface_2D': surface})

    if add_gridlines:
        grid = add_polyhedron(locs, quads, scalars=None, opacity=1.0, color_map='jet', representation='wireframe', mesh_color=mesh_color, offset=offset)
        grid.GetProperty().SetLineWidth(line_width)
        actor_dict.update({'Mesh lines 2D': grid})

        return actor_dict

    else:

        return actor_dict


def add_spherical_function(function_values, secondary_scalars=None, radius=1, scale_mesh=True, add_gridlines=False, function_name=None, colormap='jet', mesh_color='black', absolute_displacement=True, offset=[0,0,0], line_width=2):

    x_granularity = function_values.shape[0]
    y_granularity = function_values.shape[1]

    phi, theta = np.mgrid[0:np.pi:(x_granularity)*1j, 0:2 * np.pi:(y_granularity)*1j]

    coordination = np.reshape(np.arange(0,(x_granularity*y_granularity)), [x_granularity, y_granularity])

    triangles = list()

    if add_gridlines:
        quads = list()

    for x_num in range(x_granularity-1):
        for y_num in range(y_granularity-1):

            tri_1 = [coordination[x_num, y_num], coordination[x_num + 1, y_num], coordination[x_num + 1, y_num + 1]]
            triangles.append(tri_1)
            tri_2 = [coordination[x_num, y_num], coordination[x_num + 1, y_num + 1], coordination[x_num , y_num +1]]
            triangles.append(tri_2)

            if add_gridlines:
                quad_1 = [coordination[x_num, y_num], coordination[x_num + 1, y_num], coordination[x_num + 1, y_num + 1], coordination[x_num, y_num + 1]]
                quads.append(quad_1)

    triangles = np.asarray(triangles)

    if add_gridlines:
        quads = np.asarray(quads)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    locs = np.asarray([x, y, z]).T

    flat_function_values = function_values.flatten()

    if scale_mesh:
        if absolute_displacement:
            locs = locs * np.abs(flat_function_values[:, np.newaxis])
        else:
            locs = locs * flat_function_values[:, np.newaxis]

    if secondary_scalars is None:
        surface = add_polyhedron(locs, triangles, scalars=flat_function_values, opacity=1.0, color_map=colormap, offset=offset)
    else:
        if np.ndim(secondary_scalars) > 1:
            secondary_scalars = secondary_scalars.flatten()
        surface = add_polyhedron(locs, triangles, scalars=secondary_scalars, opacity=1.0, color_map=colormap, offset=offset)

    actor_dict = dict()

    if function_name is None:
        function_name = 'Surface-%i' % np.random.randint(0, 10000000)

    actor_dict.update({function_name : surface})

    if add_gridlines:
        grid = add_polyhedron(locs, quads, scalars=None, opacity=1.0, color_map=colormap, representation='wireframe', mesh_color=mesh_color, offset=offset)
        grid.GetProperty().SetLineWidth(line_width)
        grid.GetProperty().LightingOff()
        grid.GetProperty().SetRenderLinesAsTubes(True)

        if function_name is None:
            gridline_name = 'mesh lines %i' % np.random.randint(0, 10000000)
        else:
            gridline_name = '%s-gridlines' % function_name 
        actor_dict.update({gridline_name: grid})

        return actor_dict

    else:

        return actor_dict


# def rotation_matrix(axis, theta):
#     """
#     Return the rotation matrix associated with counterclockwise rotation about
#     the given axis by theta radians.
#     """
#     axis = np.asarray(axis)
#     axis = axis / math.sqrt(np.dot(axis, axis))
#     a = math.cos(theta / 2.0)
#     b, c, d = -axis * math.sin(theta / 2.0)
#     aa, bb, cc, dd = a * a, b * b, c * c, d * d
#     bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
#     return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
#                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
#                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


# def rotate_arc(arcpoints, theta=math.pi, axis=[1,0,0]):
    
#     rotated_points = np.zeros(arcpoints.shape)
#     for p, point in enumerate(arcpoints):

#         rotated_points[p,:] = np.dot(rotation_matrix(axis, theta), point)

#     return rotated_points

# def points_to_matrix(points):

#     # unique_rs = np.unique(points[:,0])
#     unique_ts = np.unique(points[:,1])
#     unique_ps = np.unique(points[:,2])



#     matrix = np.zeros((unique_ts.shape[0], unique_ps.shape[0]))

#     print(matrix.shape)

#     for point in points:

#         ti = np.where(point[1] == unique_ts)
#         pi = np.where(point[2] == unique_ps)
#         matrix[ti, pi] = point[0]

#     return matrix

if __name__ == "__main__":

    # Create a sphere
    x_granularity = 101
    y_granularity = 101

    r = 1

    phi, theta = np.mgrid[0:np.pi:(x_granularity)*1j, 0:2 * np.pi:(y_granularity)*1j]
    polar_matrix = (sph_harm(3, 4, theta, phi).real + 1 )/2


    actor_dict = dict()

    s = sph_harm(1, 4, theta, phi).real

    # # print(s)
    sp_function = add_spherical_function(polar_matrix, scale_mesh=True, absolute_displacement=True, add_gridlines=True)

    # sp_function = add_spherical_function(polar_matrix, secondary_scalars=s, scale_mesh=True, absolute_displacement=False, add_gridlines=False, mesh_color='black')

    td_function = add_2D_function(s, scale_mesh=False, absolute_displacement=True, add_gridlines=False, verbose=True)


    actor_dict.update({'spherical_function': sp_function})
    actor_dict.update({'2D_function': td_function})

    # actor_dict.update({'<points': vtk_utils.add_polydata(points, glyph_scale=1, glyph_type='sphere')})

    render(actors=actor_dict)
    # qt_utils.render_data(actor_dict=actor_dict)

