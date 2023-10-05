import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys
import os

import vtk
import LUT_utils
import interactor_utils
import physics_utils

def vtk_render(A, beta2, beta3, beta4, m2, m3, m4, theta, phi):

    r = physics_utils.calculate_r(A, beta2, m2, beta3, m3, beta4, m4, theta)

    initial_values=dict()
    initial_values.update({'A':A, 'b2':beta2, 'b3':beta3, 'b4':beta4, 'm2':m2, 'm3':m3, 'm4':m4})


    nuclear_shape = add_spherical_function(r, add_gridlines=False)

    actor_dict = dict()
    actor_dict.update({'shape': nuclear_shape})
    
    render_window = render(actors=actor_dict, initial_values=initial_values)

    return render_window

def write_gltf(source, savename='nuclear_shape.gltf', verbose=True):

    if verbose:
        print('Writing GLTF to %s' % savename)
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
        print('File written')


def ReadCubeMap(folderRoot, fileRoot, ext, key):
    """
    Read the cube map.
    :param folderRoot: The folder where the cube maps are stored.
    :param fileRoot: The root of the individual cube map file names.
    :param ext: The extension of the cube map files.
    :param key: The key to data used to build the full file name.
    :return: The cubemap texture.
    """
    # A map of cube map naming conventions and the corresponding file name
    # components.
    fileNames = {
        0: ['right', 'left', 'top', 'bottom', 'front', 'back'],
        1: ['posx', 'negx', 'posy', 'negy', 'posz', 'negz'],
        2: ['px', 'nx', 'py', 'ny', 'pz', 'nz'],
        3: ['0', '1', '2', '3', '4', '5']}
    if key in fileNames:
        fns = fileNames[key]
    else:
        print('ReadCubeMap(): invalid key, unable to continue.')
        sys.exit()
    texture = vtk.vtkTexture()
    texture.CubeMapOn()
    texture.MipmapOn()
    texture.InterpolateOn()
    # Build the file names.
    for i in range(0, len(fns)):
        fns[i] = folderRoot + fileRoot + fns[i] + ext
        if not os.path.isfile(fns[i]):
            print('Nonexistent texture file:', fns[i])
            return texture
    i = 0
    for fn in fns:
        # Read the images
        readerFactory = vtk.vtkImageReader2Factory()
        imgReader = readerFactory.CreateImageReader2(fn)
        imgReader.SetFileName(fn)

        flip = vtk.vtkImageFlip()
        flip.SetInputConnection(imgReader.GetOutputPort())
        flip.SetFilteredAxis(1)  # flip y axis
        texture.SetInputConnection(i, flip.GetOutputPort(0))
        i += 1
    return texture



class SliderProperties:
    tube_width = 0.005
    slider_length = 0.01
    slider_width = 0.01
    end_cap_length = 0.01
    end_cap_width = 0.01
    title_height = 0.025
    label_height = 0.020

    minimum_value = -2.0
    maximum_value = 2.0
    initial_value = 0.0
    tube_length = 0.18

    p1 = np.asarray([0.02, 0.05])
    p2 = p1 + np.asarray([tube_length, 0])

    title = None

    title_color = 'Black'
    label_color = 'Black'
    value_color = 'DarkSlateGray'
    slider_color = 'Red'
    selected_color = 'Lime'
    bar_color = 'DarkSlateGray'
    bar_ends_color = 'Black'


def make_slider(properties):

    colors = vtk.vtkNamedColors()

    slider = vtk.vtkSliderRepresentation2D()

    slider.SetMinimumValue(properties.minimum_value)
    slider.SetMaximumValue(properties.maximum_value)
    slider.SetValue(properties.initial_value)
    slider.SetTitleText(properties.title)

    slider.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider.GetPoint1Coordinate().SetValue(properties.p1[0], properties.p1[1])
    slider.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider.GetPoint2Coordinate().SetValue(properties.p2[0], properties.p2[1])

    slider.SetTubeWidth(properties.tube_width)
    slider.SetSliderLength(properties.slider_length)
    slider.SetSliderWidth(properties.slider_width)
    slider.SetEndCapLength(properties.end_cap_length)
    slider.SetEndCapWidth(properties.end_cap_width)
    slider.SetTitleHeight(properties.title_height)
    slider.SetLabelHeight(properties.label_height)

    # Set the color properties
    # Change the color of the title.
    slider.GetTitleProperty().SetColor(colors.GetColor3d(properties.title_color))
    # Change the color of the label.
    slider.GetTitleProperty().SetColor(colors.GetColor3d(properties.label_color))
    # Change the color of the bar.
    slider.GetTubeProperty().SetColor(colors.GetColor3d(properties.bar_color))
    # Change the color of the ends of the bar.
    slider.GetCapProperty().SetColor(colors.GetColor3d(properties.bar_ends_color))
    # Change the color of the knob that slides.
    slider.GetSliderProperty().SetColor(colors.GetColor3d(properties.slider_color))
    # Change the color of the knob when the mouse is held on it.
    slider.GetSelectedProperty().SetColor(colors.GetColor3d(properties.selected_color))
    # Change the color of the text displaying the value.
    slider.GetLabelProperty().SetColor(colors.GetColor3d(properties.value_color))

    slider_widget = vtk.vtkSliderWidget()
    slider_widget.SetRepresentation(slider)

    return slider_widget

class SliderCallback:
    def __init__(self, initial_value):
        self.value = initial_value

    def _extract_values(self, sliders):

        A = int(sliders['A'].GetRepresentation().GetValue())
        b2 = sliders['Beta 2'].GetRepresentation().GetValue()
        b3 = sliders['Beta 3'].GetRepresentation().GetValue()
        b4 = sliders['Beta 4'].GetRepresentation().GetValue()
        m2 = int(sliders['m2'].GetRepresentation().GetValue())
        m3 = int(sliders['m3'].GetRepresentation().GetValue())
        m4 = int(sliders['m4'].GetRepresentation().GetValue())
        # print(A, b2, b3, b4, m2, m3, m4)
        return A, b2, b3, b4, m2, m3, m4

    def __call__(self, caller, ev):
        slider_widget = caller
        value = slider_widget.GetRepresentation().GetValue()
        self.value = value
        A, b2, b3, b4, m2, m3, m4 = self._extract_values(sliders)
        r = physics_utils.calculate_r(A, b2, m2, b3, m3, b4, m4)
        old_shape = renderer.GetActors().GetItemAsObject(0)
        add_spherical_function(r, add_gridlines=False, original_actor=old_shape)
        renderer.Modified()

def add_sliders(interactor, renderer, initial_values=None):

    global sliders

    sliders = dict()

    sw_p = SliderProperties()

    sw_p.title = 'A'
    sw_p.minimum_value = 1
    sw_p.maximum_value = 250
    if initial_values is not None:
        sw_p.initial_value = initial_values.get('A', 100)
    else:
        sw_p.initial_value = 100

    a_slider = make_slider(sw_p)
    a_slider.SetInteractor(interactor)
    a_slider.SetAnimationModeToAnimate()
    a_slider.EnabledOn()
    a_slider.SetCurrentRenderer(renderer)
    a_slider_cb = SliderCallback(sw_p.initial_value)
    a_slider.AddObserver(vtk.vtkCommand.InteractionEvent, a_slider_cb)
    sliders.update({sw_p.title: a_slider})

    sw_p = SliderProperties()

    sw_p.p1 = sw_p.p1 + np.asarray([0.2, 0])
    sw_p.p2 = sw_p.p1 + np.asarray([sw_p.tube_length, 0])

    sw_p.title = 'Beta 2'
    if initial_values is not None:
        sw_p.initial_value = initial_values.get('b2', 0)
    else:
        sw_p.initial_value = 0
    beta_2_slider = make_slider(sw_p)
    beta_2_slider.SetInteractor(interactor)
    beta_2_slider.SetAnimationModeToAnimate()
    beta_2_slider.EnabledOn()
    beta_2_slider.SetCurrentRenderer(renderer)
    beta_2_slider_cb = SliderCallback(sw_p.initial_value)
    beta_2_slider.AddObserver(vtk.vtkCommand.InteractionEvent, beta_2_slider_cb)
    sliders.update({sw_p.title: beta_2_slider})

    sw_p.p1 = sw_p.p1 + np.asarray([0.2, 0])
    sw_p.p2 = sw_p.p1 + np.asarray([sw_p.tube_length, 0])

    sw_p.title = 'Beta 3'
    if initial_values is not None:
        sw_p.initial_value = initial_values.get('b3', 0)
    else:
        sw_p.initial_value = 0
    beta_3_slider = make_slider(sw_p)
    beta_3_slider.SetInteractor(interactor)
    beta_3_slider.SetAnimationModeToAnimate()
    beta_3_slider.EnabledOn()
    beta_3_slider.SetCurrentRenderer(renderer)
    beta_3_slider_cb = SliderCallback(sw_p.initial_value)
    beta_3_slider.AddObserver(vtk.vtkCommand.InteractionEvent, beta_3_slider_cb)
    sliders.update({sw_p.title: beta_3_slider})

    sw_p.p1 = sw_p.p1 + np.asarray([0.2, 0])
    sw_p.p2 = sw_p.p1 + np.asarray([sw_p.tube_length, 0])


    sw_p.title = 'Beta 4'
    if initial_values is not None:
        sw_p.initial_value = initial_values.get('b4', 0)
    else:
        sw_p.initial_value = 0
    beta_4_slider = make_slider(sw_p)
    beta_4_slider.SetInteractor(interactor)
    beta_4_slider.SetAnimationModeToAnimate()
    beta_4_slider.EnabledOn()
    beta_4_slider.SetCurrentRenderer(renderer)
    beta_4_slider_cb = SliderCallback(sw_p.initial_value)
    beta_4_slider.AddObserver(vtk.vtkCommand.InteractionEvent, beta_4_slider_cb)
    sliders.update({sw_p.title: beta_4_slider})

    sw_p.p1 = sw_p.p1 + np.asarray([0.2, 0])
    sw_p.p2 = sw_p.p1 + np.asarray([sw_p.tube_length, 0])




    sw_p = SliderProperties()

    sw_p.p1 = np.asarray([0.02, 0.95])
    sw_p.p2 = sw_p.p1 + np.asarray([sw_p.tube_length, 0])

    sw_p.title = 'm2'
    sw_p.minimum_value = -2
    sw_p.maximum_value = 2

    
    if initial_values is not None:
        sw_p.initial_value = initial_values.get('m2', 0)
    else:
        sw_p.initial_value = 0

    m2_slider = make_slider(sw_p)
    m2_slider.SetInteractor(interactor)
    m2_slider.SetAnimationModeToAnimate()
    m2_slider.EnabledOn()
    m2_slider.SetCurrentRenderer(renderer)
    m2_slider_cb = SliderCallback(sw_p.initial_value)
    m2_slider.AddObserver(vtk.vtkCommand.InteractionEvent, m2_slider_cb)
    sliders.update({sw_p.title: m2_slider})



    sw_p.p1 = sw_p.p1 + np.asarray([0.2, 0])
    sw_p.p2 = sw_p.p1 + np.asarray([sw_p.tube_length, 0])

    sw_p.title = 'm3'
    sw_p.minimum_value = -3
    sw_p.maximum_value = 3
    if initial_values is not None:
        sw_p.initial_value = initial_values.get('m3', 0)
    else:
        sw_p.initial_value = 0

    m3_slider = make_slider(sw_p)
    m3_slider.SetInteractor(interactor)
    m3_slider.SetAnimationModeToAnimate()
    m3_slider.EnabledOn()
    m3_slider.SetCurrentRenderer(renderer)
    m3_slider_cb = SliderCallback(sw_p.initial_value)
    m3_slider.AddObserver(vtk.vtkCommand.InteractionEvent, m3_slider_cb)
    sliders.update({sw_p.title: m3_slider})


    sw_p.p1 = sw_p.p1 + np.asarray([0.2, 0])
    sw_p.p2 = sw_p.p1 + np.asarray([sw_p.tube_length, 0])

    sw_p.title = 'm4'
    sw_p.minimum_value = -4
    sw_p.maximum_value = 4
    if initial_values is not None:
        sw_p.initial_value = initial_values.get('m4', 0)
    else:
        sw_p.initial_value = 0

    m4_slider = make_slider(sw_p)
    m4_slider.SetInteractor(interactor)
    m4_slider.SetAnimationModeToAnimate()
    m4_slider.EnabledOn()
    m4_slider.SetCurrentRenderer(renderer)
    m4_slider_cb = SliderCallback(sw_p.initial_value)
    m4_slider.AddObserver(vtk.vtkCommand.InteractionEvent, m4_slider_cb)
    sliders.update({sw_p.title: m4_slider})

    return sliders


def add_PBR(actor, metallic_factor=1, roughness_factor=0, verbose=True):

    try:
        if verbose:
            print('Adding PBR')
        actor.GetProperty().SetMetallic(metallic_factor)
        actor.GetProperty().SetRoughness(roughness_factor)
        actor.GetProperty().SetInterpolationToPBR()

        # actor.GetProperty().SetORMTexture(ormTexture)
        actor.GetProperty().SetOcclusionStrength(1)
    except Exception as e: 
        print(e)
        print('Failed to add PRB to actor')
        return actor

    return actor

def render(actors=None, background_color='White', window_size=(1200, 1200), multiview=False, add_axes=True, theta=None, use_PBR=False, initial_values=None):

    renderWindow = vtk.vtkRenderWindow()
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
  
    if multiview:
        renderWindow = multi_render(actors, background_color=background_color, window_size=window_size, render_window=renderWindow, render_interactor=renderWindowInteractor, add_axes=add_axes)


    else:

        global renderer

        renderer = vtk.vtkRenderer()
        render_camera = renderer.GetActiveCamera()
        renderWindow.SetSize(window_size)
        renderWindow.AddRenderer(renderer)

        
        renderer.SetBackground(vtk.vtkNamedColors().GetColor3d(background_color))

        renderer.SetUseDepthPeeling(1)
        renderer.SetMaximumNumberOfPeels(10)

        if use_PBR:
            cube_path = 'cubemap'
            if os.path.exists(cube_path + os.sep + 'nx.png'):
                cubemap = ReadCubeMap(cube_path, '/', '.png', 2)
                renderer.UseImageBasedLightingOn()
                renderer.SetEnvironmentTexture(cubemap, True)
            else:
                print('Could not find cubemap')
                use_PBR = False
            # renderer.SetEnvironmentCubeMap(cubemap)

        # if add_skybox:
        #     skybox = ReadCubeMap(cube_path, '/', '.png', 2)
        #     # skybox = ReadCubeMap(cube_path, '/skybox', '.jpg', 2)
        #     skybox.InterpolateOn()
        #     skybox.RepeatOff()
        #     skybox.EdgeClampOn()

        #     skyboxActor = vtk.vtkSkybox()
        #     skyboxActor.SetTexture(skybox)
        #     renderer.AddActor(skyboxActor)




        if actors is not None:
            if isinstance(actors, list):
                # print('List of actors supplied, adding list')
                for actor in actors:
                    if use_PBR:
                        renderer.AddActor(add_PBR(actor))
                    else:
                        renderer.AddActor(actor)
                    if add_axes:
                        axes = make_axes(actor, renderer)
                        renderer.AddActor(axes)

            elif isinstance(actors, dict):
                # print('List of actors supplied, adding list')
                for actor in actors.values():
                    if isinstance(actor, list):
                        for sub_actor in actor:
                            if use_PBR:
                                renderer.AddActor(add_PBR(sub_actor))
                            else:
                                renderer.AddActor(sub_actor)
                            if add_axes:
                                axes = make_axes(sub_actor, renderer)
                                renderer.AddActor(axes)
                    elif type(actor) is list:
                        for sub_actor in actor:
                            if use_PBR:
                                renderer.AddActor(add_PBR(sub_actor))
                            else:
                                renderer.AddActor(sub_actor)
                            if add_axes:
                                axes = make_axes(sub_actor, renderer)
                                renderer.AddActor(axes)
                    elif isinstance(actor, dict):
                        for sub_actor in actor.values():
                            if use_PBR:
                                renderer.AddActor(add_PBR(sub_actor))
                            else:
                                renderer.AddActor(sub_actor)
                            if add_axes:
                                axes = make_axes(sub_actor, renderer)
                                renderer.AddActor(axes)
                    elif isinstance(actor, tuple):
                        for sub_actor in actor:
                            if use_PBR:
                                renderer.AddActor(add_PBR(sub_actor))
                            else:
                                renderer.AddActor(sub_actor)
                            if add_axes:
                                axes = make_axes(sub_actor, renderer)
                                renderer.AddActor(axes)

                    else:
                        if use_PBR:
                            renderer.AddActor(add_PBR(actor))
                        else:
                            renderer.AddActor(actor)
                        if add_axes:
                            axes = make_axes(actor, renderer)
                            renderer.AddActor(axes)

        else:
            if use_PBR:
                renderer.AddActor(add_PBR(actors))
            else:
                renderer.AddActor(actors)
            if add_axes:
                axes = make_axes(actors, renderer)
                renderer.AddActor(axes)


        renderer.ResetCamera()
        render_camera.Azimuth(90)
        render_camera.SetViewUp(0,0,1)

        renderWindowInteractor.SetInteractorStyle(interactor_utils.MyInteractorStyle(renderWindowInteractor, render_camera, renderWindow))

        # current_actors = renderer.GetActors()
        # print(dir(current_actors))

        # print(current_actors.GetItemAsObject(0))
        slider = add_sliders(renderWindowInteractor, renderer, initial_values=initial_values)

    cam_orient_manipulator = vtk.vtkCameraOrientationWidget()
    cam_orient_manipulator.SetParentRenderer(renderer)
    # Enable the widget.
    cam_orient_manipulator.On()

    renderWindow.SetWindowName('Nuclear Fruit Bowl: Shape Generator')
    renderWindow.SetSize(window_size)
    renderWindow.Render()
    renderWindowInteractor.Start()



    return renderWindow

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



def multi_render(actors=None, background_color='White', window_size=(600, 600), render_window=None, render_interactor=None, share_camera=False, add_axes=False):

    border_width = 0.001


    xmins = np.asarray([0, .5, 0, .5])   +  np.asarray([0,border_width,0,border_width])
    xmaxs = np.asarray([0.5, 1, 0.5, 1]) -  np.asarray([border_width,0,border_width,0])
    ymins = np.asarray([0, 0, .5, .5])  +  np.asarray([0,0,border_width,border_width])
    ymaxs = np.asarray([0.5, 0.5, 1, 1]) -  np.asarray([border_width,border_width,0,0])

    camera_angles = [[0,0], [90,0], [180,0], [270,0]]


    if render_window is None:
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetWindowName('Nuclear Fruit Bowl: Shape Generator')
        renderWindow.SetSize(window_size)
    else:
        renderWindow = render_window

    if render_interactor is None:
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        
    else:
        renderWindowInteractor = render_interactor



    for i in range(4):
        renderer = vtk.vtkRenderer()
        renderWindow.AddRenderer(renderer)
        renderer.SetViewport(xmins[i], ymins[i], xmaxs[i], ymaxs[i])
        renderer.SetBackground(vtk.vtkNamedColors().GetColor3d(background_color))
        renderer.SetUseDepthPeeling(1)
        renderer.SetMaximumNumberOfPeels(10)

        if share_camera:
            # Share the camera between viewports.
            if i == 0:
                camera = renderer.GetActiveCamera()
                camera.Azimuth(30)
                camera.Elevation(30)
            else:
                renderer.SetActiveCamera(camera)
        else:
            camera = renderer.GetActiveCamera()
            camera.Azimuth(camera_angles[i][0])
            camera.Elevation(camera_angles[i][1])

            if i == 0:
                renderWindowInteractor.SetInteractorStyle(interactor_utils.MyInteractorStyle(renderWindowInteractor, camera, renderWindow))

        if actors is not None:
            if isinstance(actors, list):
                # print('List of actors supplied, adding list')
                for actor in actors:
                    renderer.AddActor(actor)
                    if add_axes:
                        axes = make_axes(actor, renderer)
                        renderer.AddActor(axes)

            elif isinstance(actors, dict):
                # print('List of actors supplied, adding list')
                for actor in actors.values():
                    if isinstance(actor, list):
                        for sub_actor in actor:
                            renderer.AddActor(sub_actor)
                            if add_axes:
                                axes = make_axes(sub_actor, renderer)
                                renderer.AddActor(axes)
                    elif type(actor) is list:
                        for sub_actor in actor:
                            renderer.AddActor(sub_actor)
                            if add_axes:
                                axes = make_axes(sub_actor, renderer)
                                renderer.AddActor(axes)
                    elif isinstance(actor, dict):
                        for sub_actor in actor.values():
                            renderer.AddActor(sub_actor)
                            if add_axes:
                                axes = make_axes(sub_actor, renderer)
                                renderer.AddActor(axes)
                    elif isinstance(actor, tuple):
                        for sub_actor in actor:
                            renderer.AddActor(sub_actor)
                            if add_axes:
                                axes = make_axes(sub_actor, renderer)
                                renderer.AddActor(axes)

                    else:
                        renderer.AddActor(actor)
                        if add_axes:
                            axes = make_axes(actor, renderer)
                            renderer.AddActor(axes)

        else:
            renderer.AddActor(actors)
            if add_axes:
                axes = make_axes(actors, renderer)
                renderer.AddActor(axes)

        renderer.ResetCamera()

    renderWindow.Render()


    return renderWindow


def make_axes(source_object=None,
              source_renderer=None,
              line_width=10,
              bounds=None,
              axes_type='cartesian',
              axes_units='fm',
              tick_location='outside',
              minor_ticks=False,
              axes_placement='outer',
              grid_placement='outer',
              flat_labels=True,
              sticky_axes=False,
              draw_grid_planes=False):
    """[Adds axes to an actor]

    Arguments:
        source_object {VTK Actor} -- [The actor you want the Axes to be bound to]
        source_renderer {VTK renderer} -- [the renderer used in VTK]

    Returns:
        Axes {VTK widget} -- [The axes specified]
    """

    if axes_type == 'cartesian':
        cubeAxesActor = vtk.vtkCubeAxesActor()

        if source_object is not None:
            cubeAxesActor.SetBounds(source_object.GetBounds())
        else:
            cubeAxesActor.SetBounds(bounds)

        
        if source_renderer is not None:
            cubeAxesActor.SetCamera(source_renderer.GetActiveCamera())

        cubeAxesActor.GetProperty().SetColor(0.0, 0.0, 0.0)

        cubeAxesActor.SetXTitle('X-Axis')
        cubeAxesActor.SetXUnits(axes_units)
        cubeAxesActor.GetXAxesLinesProperty().SetColor(0.0, 0.0, 0.0)
        cubeAxesActor.GetTitleTextProperty(0).SetColor(0.0, 0.0, 0.0)
        cubeAxesActor.GetLabelTextProperty(0).SetColor(0.0, 0.0, 0.0)

        cubeAxesActor.SetYTitle('Y-Axis')
        cubeAxesActor.SetYUnits(axes_units)
        cubeAxesActor.GetYAxesLinesProperty().SetColor(0.0, 0.0, 0.0)
        cubeAxesActor.GetTitleTextProperty(1).SetColor(0.0, 0.0, 0.0)
        cubeAxesActor.GetLabelTextProperty(1).SetColor(0.0, 0.0, 0.0)

        cubeAxesActor.SetZTitle('Z-Axis')
        cubeAxesActor.SetZUnits(axes_units)
        cubeAxesActor.GetZAxesLinesProperty().SetColor(0.0, 0.0, 0.0)
        cubeAxesActor.GetTitleTextProperty(2).SetColor(0.0, 0.0, 0.0)
        cubeAxesActor.GetLabelTextProperty(2).SetColor(0.0, 0.0, 0.0)

        cubeAxesActor.DrawXGridlinesOn()
        cubeAxesActor.DrawYGridlinesOn()
        cubeAxesActor.DrawZGridlinesOn()

        cubeAxesActor.SetUseBounds(True)

        cubeAxesActor.GetXAxesGridlinesProperty().SetColor(0.0, 0.0, 0.0)
        cubeAxesActor.GetYAxesGridlinesProperty().SetColor(0.0, 0.0, 0.0)
        cubeAxesActor.GetZAxesGridlinesProperty().SetColor(0.0, 0.0, 0.0)

        if vtk.VTK_MAJOR_VERSION > 5:
            if grid_placement == 'outer':
                cubeAxesActor.SetGridLineLocation(cubeAxesActor.VTK_GRID_LINES_FURTHEST)
            elif grid_placement == 'inner':
                cubeAxesActor.SetGridLineLocation(cubeAxesActor.VTK_GRID_LINES_CLOSEST)
            elif grid_placement == 'all':
                cubeAxesActor.SetGridLineLocation(cubeAxesActor.VTK_GRID_LINES_ALL)

        if minor_ticks:
            cubeAxesActor.XAxisMinorTickVisibilityOn()
            cubeAxesActor.YAxisMinorTickVisibilityOn()
            cubeAxesActor.ZAxisMinorTickVisibilityOn()

        else:
            cubeAxesActor.XAxisMinorTickVisibilityOff()
            cubeAxesActor.YAxisMinorTickVisibilityOff()
            cubeAxesActor.ZAxisMinorTickVisibilityOff()

        if tick_location == 'inside':
            cubeAxesActor.SetTickLocationToInside()
        elif tick_location == 'outside':
            cubeAxesActor.SetTickLocationToOutside()
        elif tick_location == 'both':
            cubeAxesActor.SetTickLocationToBoth()

        if axes_placement == 'outer':
            cubeAxesActor.SetFlyModeToOuterEdges()
            cubeAxesActor.SetTickLocationToOutside()
        elif axes_placement == 'inner':
            cubeAxesActor.SetFlyModeToClosestTriad()
        elif axes_placement == 'furthest':
            cubeAxesActor.SetFlyModeToFurthestTriad()
        elif axes_placement == 'all':
            cubeAxesActor.SetFlyModeToStaticEdges()

        # cubeAxesActor.SetUse2DMode(flat_labels)
        # cubeAxesActor.SetUseTextActor3D(True)
        cubeAxesActor.SetStickyAxes(sticky_axes)
        cubeAxesActor.SetCenterStickyAxes(False)

        cubeAxesActor.GetProperty().SetLineWidth(line_width)
        # cubeAxesActor.Update()
        cubeAxesActor.GetProperty().RenderLinesAsTubesOn()
        # cubeAxesActor.GetProperty().SetEdgeVisibility(True)
        cubeAxesActor.GetProperty().SetInterpolationToPhong()
        cubeAxesActor.GetProperty().VertexVisibilityOn()
        cubeAxesActor.SetUse2DMode(True)

        x_properties = cubeAxesActor.GetTitleTextProperty(0)
        x_properties.BoldOn()
        x_properties.ItalicOn()
        x_properties.SetFontSize(20)
        x_properties.SetLineOffset(50)
        x_properties.SetVerticalJustificationToTop()

        y_properties = cubeAxesActor.GetTitleTextProperty(1)
        y_properties.BoldOn()
        y_properties.ItalicOn()
        y_properties.SetFontSize(20)
        y_properties.SetLineOffset(50)

        z_properties = cubeAxesActor.GetTitleTextProperty(2)
        z_properties.BoldOn()
        z_properties.ItalicOn()
        z_properties.SetFontSize(20)
        z_properties.SetLineOffset(50)
    

        if draw_grid_planes:
            cubeAxesActor.DrawXGridpolysOn()
            cubeAxesActor.DrawYGridpolysOn()
            cubeAxesActor.DrawZGridpolysOn()

        cubeAxesActor.SetUseTextActor3D(2)
        cubeAxesActor.SetUseTextActor3D(1)
        cubeAxesActor.SetUseTextActor3D(0)
        cubeAxesActor.SetUse2DMode(False)

        cubeAxesActor.Modified()


        return cubeAxesActor

    elif axes_type == 'polar':
        # pole = [0, 0, 0]
        polaxes = vtk.vtkPolarAxesActor()
        # polaxes.SetPole(pole)

        # polaxes.SetMaximumRadius(80)
        polaxes.SetMinimumAngle(0.)
        polaxes.SetMaximumAngle(360.)
        polaxes.SetSmallestVisiblePolarAngle(1.)
        polaxes.SetUse2DMode(flat_labels)

        polaxes.SetAutoSubdividePolarAxis(True)
        if source_renderer is not None:
            polaxes.SetCamera(source_renderer.GetActiveCamera())
        polaxes.SetPolarLabelFormat("%6.1f")
        polaxes.GetSecondaryRadialAxesProperty().SetColor(1., 0., 0.)
        polaxes.GetSecondaryRadialAxesTextProperty().SetColor(0., 0., 1.)
        polaxes.GetPolarArcsProperty().SetColor(1., 0., 0.)
        polaxes.GetSecondaryPolarArcsProperty().SetColor(0., 0., 1.)
        polaxes.GetPolarAxisProperty().SetColor(0., 0, 0.)
        polaxes.GetPolarAxisTitleTextProperty().SetColor(0., 0., 0.)
        polaxes.GetPolarAxisLabelTextProperty().SetColor(0., 0., 0.)
        polaxes.SetEnableDistanceLOD(True)

        if minor_ticks:
            polaxes.SetAxisMinorTickVisibility(1)

        polaxes.SetScreenSize(6.)

        if source_object is not None:
            object_bounds = np.asarray(source_object.GetBounds())
            del_x = object_bounds[1] - object_bounds[0]
            del_y = object_bounds[3] - object_bounds[2]

            max_bound = max(del_x, del_y)
            polaxes.SetMaximumRadius(max_bound/2)

        else:
            polaxes.SetBounds(bounds)

        # polaxes.SetStickyAxes(True)
        # polaxes.SetCenterStickyAxes(False)
        return polaxes

def add_polyhedron(vertices, faces, labels=None, offset=[0, 0, 0], scalars=None, secondary_offset=None, original_actor=None, rotation=None, generate_normals=False, opacity=1.0, verbose=False, mesh_color='black', color_map='viridis', c_range=None, representation='surface', interpolate_scalars=True):

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

    if generate_normals:
        normal_filter = vtk.vtkPolyDataNormals()
        normal_filter.SetInputData(polydata)
        normal_filter.Update()

    if original_actor is None:

        # print(normal_filter.GetOutput(), scalars.shape)
        # Create a mapper and actor
        mapper = vtk.vtkDataSetMapper()
        if generate_normals:
            mapper.SetInputData(normal_filter.GetOutput())
        else:
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

    else:

        if generate_normals:
            normal_filter = vtk.vtkPolyDataNormals()
            normal_filter.SetInputData(polydata)
            normal_filter.Update()
            original_actor.GetMapper().SetInputData(normal_filter.GetOutput())

        else:
            original_actor.GetMapper().SetInputData(polydata)

        

        if scalars is not None:

            if c_range is None:
                c_range = (np.min(scalars), np.max(scalars))
                
            lut = LUT_utils.make_LUT(colormap=color_map, c_range=c_range, nan_color=(0,1,0,1))

            original_actor.GetMapper().SetLookupTable(lut)
            original_actor.GetMapper().SetScalarRange(c_range)
            
            cur_color_data = vtk.vtkUnsignedCharArray()
            cur_color_data.SetNumberOfComponents(3)
            cur_color_data.SetName("Colors")
            for val in scalars:
                col = [0., 0., 0.]
                lut.GetColor(val, col)
                col = [int(c * 255) for c in col]
                cur_color_data.InsertNextTuple3(col[0], col[1], col[2])

            original_actor.GetMapper().GetInput().GetPointData().SetScalars(cur_color_data)
            original_actor.GetMapper().SetInterpolateScalarsBeforeMapping(interpolate_scalars)

        original_actor.GetMapper().Modified()
        original_actor.Modified()

        return original_actor




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
    z = function_values.flatten() * scale

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


def add_spherical_function(function_values, secondary_scalars=None, radius=1, scale_mesh=True, add_gridlines=False, function_name=None, colormap='jet', mesh_color='black', absolute_displacement=True, offset=[0,0,0], line_width=2, original_actor=None):

    function_values = function_values.T
    
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
        surface = add_polyhedron(locs, triangles, scalars=flat_function_values, opacity=1.0, color_map=colormap, offset=offset, original_actor=original_actor)
    else:
        if np.ndim(secondary_scalars) > 1:
            secondary_scalars = secondary_scalars.flatten()
        surface = add_polyhedron(locs, triangles, scalars=secondary_scalars, opacity=1.0, color_map=colormap, offset=offset, original_actor=original_actor)

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



if __name__ == "__main__":

    # Create a sphere

    # Make mesh of thetas and phis
    phi, theta = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]

    # initialise deformations for a spherical nucleus with mass A=100
    beta2 = 0.5
    beta3 = 0
    beta4 = 0
    A = 100

    m2 = 0
    m3 = 0
    m4 = 0

    r = physics_utils.calculate_r(A, beta2, m2, beta3, m3, beta4, m4, theta)

    initial_values=dict()
    initial_values.update({'A':A, 'b2':beta2, 'b3':beta3, 'b4':beta4, 'm2':m2, 'm3':m3, 'm4':m4})

    # # print(s)
    sp_function = add_spherical_function(r, scale_mesh=True, absolute_displacement=True, add_gridlines=True)

    td_function = add_2D_function(r, scale_mesh=False, absolute_displacement=True, x_range=[0,10], y_range=[0,10], add_gridlines=True, verbose=False, offset=[8,-5,-55], scale=10, mesh_color='black')
    actor_dict = dict()

    actor_dict.update({'spherical_function': sp_function})
    actor_dict.update({'2D_function': td_function})


    render(actors=actor_dict, initial_values=initial_values)

