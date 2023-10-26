import warnings
import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys
import os
import config


from vtk.util import numpy_support
import PIL

import vtk
import LUT_utils
import interactor_utils
import physics_utils


def vtk_render(A, beta2, beta3, beta4, m2, m3, m4, theta, phi, secondary_scalar=None):


    # print(globals()["savename"])
    r = physics_utils.calculate_r(A, beta2, m2, beta3, m3, beta4, m4, theta)

    initial_values=dict()
    initial_values.update({'A':A, 'b2':beta2, 'b3':beta3, 'b4':beta4, 'm2':m2, 'm3':m3, 'm4':m4})

    print('press \'r\' to reset camera')
    print('press \'o\' to toggle orthogonal projection')
    print('press \'w\' to render as wireframe')
    print('press \'s\' to render as surface')
    print('press \'i\' to change display type (flat, smooth, reflective)')
    print('press \'e\' or \'q\' to exit')

    nuclear_shape = add_spherical_function(r, add_gridlines=False, secondary_scalars=secondary_scalar, colormap=config.colormap)

    # nuclear_shape = add_textures_to_actor(nuclear_shape[list(nuclear_shape.keys())[0]], material='treadplate')

    actor_dict = dict()
    actor_dict.update({'shape': nuclear_shape})
    
    render_window = render(actors=actor_dict, initial_values=initial_values, secondary_scalar=secondary_scalar)

    return render_window

def write_gltf(source, verbose=True):

    savename = config.savename

    if verbose:
        print('Writing GLTF to %s' % savename)
    try:
        exporter = vtk.vtkGLTFExporter()
    except AttributeError:
        print('Gltf exporting is not supported in your version of VTK, try updating')
    exporter.SetInput(source)
    exporter.InlineDataOn()
    exporter.SaveNormalOn()
    exporter.SetFileName(savename)
    exporter.Update()
    exporter.Write()
    if verbose:
        print('File written')



def colorbar(actor,
            interactor=None,
            title='Radius',
            orientation='vertical',
            return_widget=True,
            total_ticks=10,
            background=True,
            opacity=1.0):
    """[Adds a colorbar to the interactor]

    Arguments:
        interactor {[VTK Interactor]} -- [the interactor used by VTK]
        title {[string]} -- [the title of the colorbar]
        mapper {[VTK mapper]} -- [the VTK mapper used to grab color values]

    Keyword Arguments:
        orientation {str} -- [orientation of the colorbar] (default: {'vertical'})
        return_widget {bool} -- [returns the colorbar as a widget instead] (default: {True})

    Returns:
        [type] -- [description]
    """
    print('adding colorbar')

    if type(actor) == vtk.vtkScalarBarActor:

        scalar_bar = actor

    else:

        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(actor.GetMapper().GetLookupTable())

        # print(scalar_bar)

        # print(sorted(dir(scalar_bar)))

        scalar_bar.GetLabelTextProperty().SetColor(0.0, 0.0, 0.0)
        scalar_bar.GetLabelTextProperty().SetFontFamilyToArial()
        scalar_bar.GetLabelTextProperty().ItalicOff()
        # scalar_bar.GetLabelTextProperty().ShadowOff()
        # scalar_bar.GetLabelTextProperty().FrameOn()
        scalar_bar.GetLabelTextProperty().SetShadowOffset(0,0)
        scalar_bar.GetLabelTextProperty().GetShadowColor((1,1,1))#.SetValue(1,1,1)
        # scalar_bar.GetLabelTextProperty().SetBackgroundOpacity(0.5)
        # scalar_bar.GetLabelTextProperty().SetBackgroundColor(1.0, 1.0, 1.0)
        # print('SHADOW', scalar_bar.GetLabelTextProperty().GetShadow())

        scalar_bar.SetTitle(title)
        scalar_bar.GetTitleTextProperty().SetColor(0.0, 0.0, 0.0)
        # scalar_bar.SetUnconstrainedFontSize(40)
        scalar_bar.GetTitleTextProperty().SetFontFamilyToArial()
        scalar_bar.GetTitleTextProperty().ShadowOff()
        scalar_bar.GetTitleTextProperty().ItalicOff()
        scalar_bar.AnnotationTextScalingOn()
        scalar_bar.SetVerticalTitleSeparation(10)
        scalar_bar.UseOpacityOff()

        scalar_bar.Modified()

        # print(dir(scalar_bar))
        if orientation == 'Horizontal':
            scalar_bar.SetOrientationToHorizontal()
        else:
            scalar_bar.SetOrientationToVertical()
        scalar_bar.SetWidth(0.1)
        # scalar_bar.SetHeight(0.1)
        scalar_bar.SetVisibility(1)
        scalar_bar.SetNumberOfLabels(total_ticks)
        scalar_bar.UseOpacityOn()
        if background:
            scalar_bar.DrawBackgroundOn()
            scalar_bar.GetBackgroundProperty().SetOpacity(opacity)
        # scalar_bar.SetWidth(10)
        scalar_bar.SetBarRatio(0.85)


    # create the scalar_bar_widget
    if return_widget:

        global scalar_bar_widget

        scalar_bar_widget = vtk.vtkScalarBarWidget()
        scalar_bar_widget.SetInteractor(interactor)
        scalar_bar_widget.SetScalarBarActor(scalar_bar)
        scalar_bar_widget.On()
        scalar_bar_widget.ResizableOn()


        # print(scalar_bar, scalar_bar_widget)

        return scalar_bar_widget

    return scalar_bar



def read_texture(filename, file_type='jpg', useSRGB=False):

    if file_type == '.png':
        color = vtk.vtkPNGReader()
    if file_type == '.jpg':
        color = vtk.vtkJPEGReader()
    elif file_type == '.tiff':
        warnings.warn('TIFF images may produce odd effects, JPEGs or PNGs are recommended')
        color = vtk.vtkTIFFReader()
    elif file_type == '.tif':
        warnings.warn('TIFF images may produce odd effects, JPEGs or PNGs are recommended')
        color = vtk.vtkTIFFReader()
    color.SetFileName(filename)
    color_texture = vtk.vtkTexture()
    color_texture.SetInputConnection(color.GetOutputPort())

    if useSRGB:
        color_texture.UseSRGBColorSpaceOn()

    return color_texture


def read_texture_directory(folder_path, color=None, texture_size=[1024,1024], fallback_ORM=[128,0,0], fallback_normals=[128,128, 255], fallback_height=128):

    file_type = '.jpg'

    if color is None:

        albedofile = folder_path + os.sep + "albedo" + file_type

    else:

        albedofile = folder_path + os.sep + color + file_type


    if not os.path.isfile(albedofile):

        file_type = '.png'

        if color is None:

            albedofile = folder_path + os.sep + "albedo" + file_type

        else:

            albedofile = folder_path + os.sep + color + file_type

    if not os.path.isfile(albedofile):


        file_type = '.tif'

        if color is None:

            albedofile = folder_path + os.sep + "albedo" + file_type

        else:

            albedofile = folder_path + os.sep + color + file_type

    if not os.path.isfile(albedofile):

        print('can\'t find textures in directory:', folder_path)

    albedo_texture = read_texture(albedofile, useSRGB=True, file_type=file_type)


    normalfile = folder_path + os.sep + "normal" + file_type
    heightfile = folder_path + os.sep + "height" + file_type
    emissivefile = folder_path + os.sep + "emissive" + file_type
    ormfile =  folder_path + os.sep + "orm" + file_type

    a = np.array(PIL.Image.open(albedofile))
    if np.ndim(a) == 3:
        a = a[:,:,0]
    texture_size = a.shape

    if os.path.isfile(ormfile):

        orm_texture = read_texture(ormfile, file_type=file_type)

    else:

        print('Can\'t find ORM file, assuming seperate Occlusion, Roughness & Metalicity')

        occlusionfile = folder_path + os.sep + "ao" + file_type
        roughnessfile = folder_path + os.sep + "roughness" + file_type
        specularfile = folder_path + os.sep + "specular" + file_type
        metalfile = folder_path + os.sep + "metallic" + file_type

        if os.path.isfile(occlusionfile):
            o = np.array(PIL.Image.open(occlusionfile))
            if np.ndim(o) == 3:
                o = o[:,:,0]
        else:
            print('Occlusion file not found, using fallback value: %f' % fallback_ORM[0])
            o = np.ones(texture_size) * fallback_ORM[0]

        if os.path.isfile(roughnessfile):
            r = np.array(PIL.Image.open(roughnessfile))
            if np.ndim(r) == 3:
                r = r[:,:,0]
        elif os.path.isfile(specularfile):
            r = 1 - np.array(PIL.Image.open(specularfile))
            if np.ndim(r) == 3:
                r = r[:,:,0]
                print(r)
        else:
            print('Roughness file not found, using fallback value: %f' % fallback_ORM[1])
            r = np.ones(texture_size) * fallback_ORM[1]

        if os.path.isfile(metalfile):
            m = np.array(PIL.Image.open(metalfile))
            if np.ndim(m) == 3:
                m = m[:,:,0]
        else:
            print('Metalicity file not found, using fallback value: %f' % fallback_ORM[2])
            m = np.ones(texture_size) * fallback_ORM[2]

        orm = np.dstack([o,r,m])

        orm_image = PIL.Image.fromarray(orm.astype(np.uint8))
        print('saving new ORM texture to %s' % ormfile)
        orm_image.save(ormfile)

        orm_texture = read_texture(ormfile, file_type=file_type)


    if os.path.isfile(normalfile):
        normal_texture = read_texture(normalfile, file_type=file_type)
    else:
        print('Normal file not found, creating fallback')
        normal_array = np.ones([*a.shape, 3]) * fallback_normals
        normal_image = PIL.Image.fromarray(normal_array.astype(np.uint8))
        normal_image.save(normalfile)
        normal_texture = read_texture(normalfile, file_type=file_type)


    texture_dict = dict()

    texture_dict.update({'albedo':albedo_texture})
    texture_dict.update({'normal':normal_texture})
    texture_dict.update({'ORM':orm_texture})

    if os.path.isfile(heightfile):
        height_texture = read_texture(heightfile, useSRGB=True, file_type=file_type)
        texture_dict.update({'height':height_texture})
    else:
        pass

    if os.path.isfile(emissivefile):
        emissive_texture = read_texture(emissivefile, useSRGB=True, file_type=file_type)
        texture_dict.update({'emissive':emissive_texture})
    else:
        pass

    return texture_dict

def generate_texture_coords(uResolution, vResolution, pd):
    """
    Generate u, v texture coordinates on a parametric surface.
    :param uResolution: u resolution
    :param vResolution: v resolution
    :param pd: The polydata representing the surface.
    :return: The polydata with the texture coordinates added.
    """
    print('Can\'t find texture coordinates, making new ones')
    numPts = pd.GetNumberOfPoints()

    if ((uResolution is None) or (vResolution is None)):
        limit = int(np.floor(np.sqrt(numPts)))
        uResolution = limit
        vResolution = limit

    elif numPts < (uResolution * vResolution):
        limit = int(np.floor(np.sqrt(numPts)))
        warnings.warn('Texture coords set too high, setting both to: %i' % limit)
        uResolution = limit
        vResolution = limit

    u0 = 1.0
    v0 = 0.0
    du = 1.0 / (uResolution - 1)
    dv = 1.0 / (vResolution - 1)

    tCoords = vtk.vtkFloatArray()
    tCoords.SetNumberOfComponents(2)
    tCoords.SetNumberOfTuples(numPts)
    tCoords.SetName('Texture Coordinates')
    ptId = 0
    u = u0
    for i in range(0, uResolution):
        v = v0
        for j in range(0, vResolution):
            tc = [u, v]
            # print(ptId, tc)
            tCoords.SetTuple(ptId, tc)
            v += dv
            ptId += 1
        u -= du
    pd.GetPointData().SetTCoords(tCoords)
    return pd, tCoords


def add_textures_to_actor(actor, material=None, texture_dir='textures', color=None, occlusion=0.5, roughness=1, metallic=1, normal_scale=1, height_scale=0.01, emissive=(1,1,1), uResolution=None, vResolution=None, use_heightmap=False, method='vtk', texture_scale=(1,1,1), verbose=False):

    fallback_ORM = np.asarray([occlusion, roughness, metallic]) * 255

    # actor.GetProperty().SetInterpolationToPBR()

    tex = actor.GetTexture()

    if tex is None:
        print('reading textures from', texture_dir + os.sep + material)
        if material is not None:
            texture_dict = read_texture_directory(texture_dir + os.sep + material, fallback_ORM=fallback_ORM, color=color)
        else:
            texture_dict = read_texture_directory(texture_dir, fallback_ORM=fallback_ORM, color=color)

    # if (('height' in texture_dict) and (use_heightmap)):

    #     if material is not None:
    #         filepath = texture_dir + material + '/height.jpg'
    #     else:
    #         filepath = texture_dir + 'height.jpg'
    #     reader = vtk.vtkJPEGReader()
    #     if not os.path.isfile(filepath):
    #         if material is not None:
    #             filepath = texture_dir + material + '/height.png'
    #         else:
    #             filepath = texture_dir + 'height.png'
    #         reader = vtk.vtkPNGReader()
    #         if not os.path.isfile(filepath):
    #             if material is not None:
    #                 filepath = texture_dir + material + '/height.tiff'
    #             else:
    #                 filepath = texture_dir + 'height.tiff'
    #             reader = vtk.vtkTIFFReader()

    #     if verbose:

    #         print('Height displacement enabled, using %s' % filepath)


    #     reader.SetFileName(filepath)
    #     reader.Update()

    #     image_extents = np.asarray(reader.GetDataExtent())
    #     image_size = np.asarray([image_extents[1] - image_extents[0], image_extents[3] - image_extents[2], image_extents[5] - image_extents[4]])



    #     current_tocoords = actor.GetMapper().GetInput().GetPointData().GetTCoords()

    #     # print('Current Coordinates', current_tocoords)

    #     if current_tocoords is not None:

    #         mapper = actor.GetMapper()

    #         probe_points = vtk.vtkPoints()

    #         # print(current_tocoords)

    #         # print(image_size)

    #         probe_points.SetNumberOfPoints(current_tocoords.GetNumberOfTuples())

    #         # print(probe_points.GetDataType())

    #         # print(dir(probe_points))

    #         np_coords = numpy_support.vtk_to_numpy(current_tocoords)
    #         n_0 = np.zeros([current_tocoords.GetNumberOfTuples(),1])
            
    #         new_coords = numpy_support.numpy_to_vtk(np.hstack([np_coords, n_0])  * image_size)

    #         # probe_points.SetNumberOfComponents(new_coords.GetNumberOfComponents())
    #         probe_points.SetData(new_coords)
    #         probe_poly = vtk.vtkPolyData()
    #         probe_poly.SetPoints(probe_points)

    #     else:

    #         pd = generate_texture_coords(uResolution, vResolution, actor.GetMapper().GetInput())

    #         tcoords = pd.GetPointData().GetTCoords()

    #         probe_points = vtk.vtkPoints()
    #         probe_points.SetNumberOfPoints(tcoords.GetNumberOfValues())

    #         np_coords = numpy_support.vtk_to_numpy(tcoords)
    #         n_0 = np.zeros([tcoords.GetNumberOfTuples(),1])

    #         probe_points.SetData(numpy_support.numpy_to_vtk(np.hstack([np_coords, n_0]) * image_size))

    #         probe_poly = vtk.vtkPolyData()
    #         probe_poly.SetPoints(probe_points)

    #         current_tocoords = probe_poly

    #     probes = vtk.vtkProbeFilter()
    #     probes.SetSourceData(reader.GetOutput())
    #     probes.SetInputData(probe_poly)
    #     probes.Update()

    #     actor.GetMapper().GetInput().GetPointData().SetScalars(probes.GetOutput().GetPointData().GetScalars())

    #     warp = vtk.vtkWarpScalar()
    #     warp.SetInputData(actor.GetMapper().GetInput())
    #     warp.SetScaleFactor(height_scale)
    #     warp.Update()

    #     mapper = vtk.vtkPolyDataMapper()
    #     mapper.SetInputConnection(warp.GetOutputPort())
    #     mapper.GetInput().GetPointData().SetScalars(None)

    #     smoothing_passes = 1

    #     if smoothing_passes is not None:

    #         if verbose:
    #             print('Smoothing mesh, may take a while')

    #         smooth_loop = vtk.vtkSmoothPolyDataFilter()
    #         smooth_loop.SetNumberOfIterations(smoothing_passes)
    #         smooth_loop.SetRelaxationFactor(0.5)
    #         smooth_loop.BoundarySmoothingOn()
    #         smooth_loop.SetInputData(mapper.GetInput())
    #         smooth_loop.Update()
    #         mapper = vtk.vtkPolyDataMapper()

    #         mapper.SetInputConnection(smooth_loop.GetOutputPort())

    #     # tcoords_np = numpy_support.vtk_to_numpy(current_tocoords)

    #     # print(tcoords_np.shape)

    #     # plt.figure()
    #     # plt.scatter(tcoords_np[:,0], tcoords_np[:,1])
    #     # plt.show()


    # else:

    #     # # print(actor)

    #     # current_tocoords = actor.GetMapper().GetInput().GetPointData().GetTCoords()
    #     # tcoords_np = numpy_support.vtk_to_numpy(current_tocoords)

    #     # # print(tcoords_np)

    #     # plt.figure()
    #     # plt.scatter(tcoords_np[:,0], tcoords_np[:,1])
    #     # plt.show()

    #     # print('current texture coordinates', current_tocoords)

    mapper = actor.GetMapper()

    current_tcoords = mapper.GetInput().GetCellData().GetTCoords()

    if current_tcoords is None:

        pd, tcords = generate_texture_coords(uResolution, vResolution, mapper.GetInput())

        # tangents = vtk.vtkPolyDataTangents()
        # tangents.SetComputePointTangents(True)
        # tangents.SetComputeCellTangents(True)
        # tangents.SetInputData(pd)
        # tangents.Update()

        mapper.SetInputData(pd)

        # mapper.SetInputConnection(tangents.GetOutputPort())


        mapper.GetInput().GetPointData().SetTCoords(tcords)
        mapper.GetInput().GetCellData().SetTCoords(tcords)
        mapper.Modified()

        # print(tcords)

    else: 

        pass


    # actor.SetMapper(mapper)

    if tex is None:


        actor.GetProperty().SetRoughness(roughness)
        actor.GetProperty().SetMetallic(metallic)
        actor.GetProperty().SetOcclusionStrength(occlusion)
        actor.GetProperty().SetNormalScale(normal_scale)
        actor.GetProperty().SetEmissiveFactor(emissive)

        actor.SetTexture(texture_dict['albedo'])
        # actor.GetProperty().SetBaseColorTexture(texture_dict['albedo'])
        actor.GetProperty().SetNormalTexture(texture_dict['normal'])
        actor.GetProperty().SetORMTexture(texture_dict['ORM'])

        if 'emissive' in texture_dict:

            actor.GetProperty().SetEmissiveTexture(texture_dict['emissive'])

    return actor


def ReadCubeMap(folderRoot, fileRoot, ext, key, flip_axis=1):
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

        if flip_axis is not None:
            flip = vtk.vtkImageFlip()
            flip.SetInputConnection(imgReader.GetOutputPort())
            flip.SetFilteredAxis(flip_axis)  # flip y axis
            texture.SetInputConnection(i, flip.GetOutputPort(0))
        else:
            texture.SetInputConnection(i, imgReader.GetOutputPort())
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

    minimum_value = -3.0
    maximum_value = 3.0
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

    initial_values=None


def make_slider(properties, slider_name=None):

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

    if slider_name is None:
            slider_name = 'slider-%i' % np.random.randint(10000,9000000)
    # print(dir(slider_widget))
    slider_widget.SetObjectName(slider_name)

    return slider_widget

class SliderCallback:
    def __init__(self, initial_value=None, secondary_scalar=None):
        self.value = initial_value
        self.secondary_scalar = secondary_scalar

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
        actors = renderer.GetActors() 

        add_spherical_function(r, add_gridlines=False, original_actor=old_shape, secondary_scalars=self.secondary_scalar, colormap=config.colormap)
        
        actors.InitializeObjectBase()
        actors.InitTraversal()
        source_actor = None

        total_actors = actors.GetNumberOfItems()
        for j in range(total_actors):
            actor = actors.GetNextActor()
            # print(type(actor), total_actors)
            if isinstance(actor, vtk.vtkOpenGLActor):
                sourcename = actor.GetObjectName()
                # print('sourcename', sourcename)
                if sourcename is not None:
                    if sourcename.startswith('surface'):
                        source_actor = actor
            if source_actor is not None:
                if isinstance(actor, vtk.vtkCubeAxesActor):
                    actor.SetBounds(source_actor.GetBounds())


        # _ = add_textures_to_actor(source_actor, material='rock')

        renderer.Modified()

        scalar_bar_widget.GetScalarBarActor().SetLookupTable(source_actor.GetMapper().GetLookupTable())
        # .Modified()
        scalar_bar_widget.Modified()

        # print(dir(scalar_bar_widget))




def export_button_callback(widget, event, savename='shape.gltf'):
    value = widget.GetRepresentation().GetState()
    renwin = widget.GetCurrentRenderer().GetRenderWindow()
  
    # print("Button pressed!", value)
    write_gltf(renwin)
    return

def reset_button_callback(widget, event):
    value = widget.GetRepresentation().GetState()

    reset_sliders(renderer)
    return


def add_export_button(interactor, renderer):

    r1 = vtk.vtkPNGReader()
    r1.SetFileName('icons/save.png')
    r1.Update()


    r2 = vtk.vtkPNGReader()
    r2.SetFileName('icons/save.png')
    r2.Update()

    buttonRepresentation = vtk.vtkTexturedButtonRepresentation2D() 
    buttonRepresentation.SetNumberOfStates(2)

    buttonRepresentation.SetButtonTexture(0, r1.GetOutput())
    buttonRepresentation.SetButtonTexture(1, r2.GetOutput())

    buttonWidget =   vtk.vtkButtonWidget()
    buttonWidget.SetInteractor(interactor)
    buttonWidget.SetRepresentation(buttonRepresentation)

    # // Place the widget. Must be done after a render so that the
    # // viewport is defined..
    # // Here the widget placement is in normalized display coordinates.
    upperRight = vtk.vtkCoordinate()
    upperRight.SetCoordinateSystemToNormalizedDisplay()
    upperRight.SetValue(0.05, 0.05)

    bds = np.zeros(6)
    sz = 50.0
    bds[0] = upperRight.GetComputedDisplayValue(renderer)[0] - sz
    bds[1] = bds[0] + sz
    bds[2] = upperRight.GetComputedDisplayValue(renderer)[1] - sz
    bds[3] = bds[2] + sz
    bds[4] = bds[5] = 0.0

    # // Scale to 1, default is .5
    buttonRepresentation.SetPlaceFactor(1)
    buttonRepresentation.PlaceWidget(bds)

    buttonWidget.AddObserver(vtk.vtkCommand.StateChangedEvent, export_button_callback)

    buttonWidget.On()

    return buttonWidget


def add_reset_button(interactor, renderer):

    r1 = vtk.vtkPNGReader()
    r1.SetFileName('icons/reset.png')
    r1.Update()


    r2 = vtk.vtkPNGReader()
    r2.SetFileName('icons/reset.png')
    r2.Update()

    buttonRepresentation = vtk.vtkTexturedButtonRepresentation2D() 
    buttonRepresentation.SetNumberOfStates(2)

    buttonRepresentation.SetButtonTexture(0, r1.GetOutput())
    buttonRepresentation.SetButtonTexture(1, r2.GetOutput())

    buttonWidget =   vtk.vtkButtonWidget()
    buttonWidget.SetInteractor(interactor)
    buttonWidget.SetRepresentation(buttonRepresentation)

    # // Place the widget. Must be done after a render so that the
    # // viewport is defined..
    # // Here the widget placement is in normalized display coordinates.
    upperRight = vtk.vtkCoordinate()
    upperRight.SetCoordinateSystemToNormalizedDisplay()
    upperRight.SetValue(0.05, 0.1)

    bds = np.zeros(6)
    sz = 50.0
    bds[0] = upperRight.GetComputedDisplayValue(renderer)[0] - sz
    bds[1] = bds[0] + sz
    bds[2] = upperRight.GetComputedDisplayValue(renderer)[1] - sz
    bds[3] = bds[2] + sz
    bds[4] = bds[5] = 0.0

    # // Scale to 1, default is .5
    buttonRepresentation.SetPlaceFactor(1)
    buttonRepresentation.PlaceWidget(bds)

    buttonWidget.AddObserver(vtk.vtkCommand.StateChangedEvent, reset_button_callback)

    buttonWidget.On()

    return buttonWidget

def reset_sliders(renderer):

    sliders = config.sliders

    sliders['A'].GetRepresentation().SetValue(config.A)
    sliders['A'].Modified()
    sliders['Beta 2'].GetRepresentation().SetValue(config.B2)
    sliders['Beta 2'].Modified()
    sliders['Beta 3'].GetRepresentation().SetValue(config.B3)
    sliders['Beta 3'].Modified()
    sliders['Beta 4'].GetRepresentation().SetValue(config.B4)
    sliders['Beta 4'].Modified()

    sliders['m2'].GetRepresentation().SetValue(config.M2)
    sliders['m2'].Modified()
    sliders['m3'].GetRepresentation().SetValue(config.M3)
    sliders['m3'].Modified()
    sliders['m4'].GetRepresentation().SetValue(config.M4)
    sliders['m4'].Modified()
    
    renderer.Render()


def add_sliders(interactor, renderer, initial_values=None, secondary_scalar=None):

    global sliders

    sliders = dict()

    sw_p = SliderProperties()
    sw_p.initial_values = initial_values

    sw_p.p1 = sw_p.p1 + np.asarray([0.05, 0])
    sw_p.p2 = sw_p.p1 + np.asarray([sw_p.tube_length, 0])

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
    a_slider_cb = SliderCallback(sw_p.initial_value, secondary_scalar=secondary_scalar)
    a_slider.AddObserver(vtk.vtkCommand.InteractionEvent, a_slider_cb)
    sliders.update({sw_p.title: a_slider})

    sw_p = SliderProperties()

    sw_p.p1 = sw_p.p1 + np.asarray([0.25, 0])
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
    beta_2_slider_cb = SliderCallback(sw_p.initial_value, secondary_scalar=secondary_scalar)
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
    beta_3_slider_cb = SliderCallback(sw_p.initial_value, secondary_scalar=secondary_scalar)
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
    beta_4_slider_cb = SliderCallback(sw_p.initial_value, secondary_scalar=secondary_scalar)
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
    m2_slider_cb = SliderCallback(sw_p.initial_value, secondary_scalar=secondary_scalar)
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
    m3_slider_cb = SliderCallback(sw_p.initial_value, secondary_scalar=secondary_scalar)
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
    m4_slider_cb = SliderCallback(sw_p.initial_value, secondary_scalar=secondary_scalar)
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

def render(actors=None, background_color='White', window_size=(1200, 1200), multiview=False, add_axes=True, add_colorbar=True, secondary_scalar=None, theta=None, use_PBR=True, initial_values=None):

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
            use_hdr=False
            if use_hdr: 
                print('using HDR')
                reader = vtk.vtkHDRReader()
                reader.SetFileName(cube_path + os.sep + 'lab.hdr')
                texture = vtk.vtkTexture()
                texture.SetColorModeToDirectScalars()
                texture.SetInputConnection(reader.GetOutputPort())
                texture.MipmapOn()
                texture.InterpolateOn()
                renderer.UseSphericalHarmonicsOn()
                renderer.SetEnvironmentTexture(texture, True)
                renderer.UseImageBasedLightingOn()

            else:
                if os.path.exists(cube_path + os.sep + 'nx.png'):
                    cubemap = ReadCubeMap(cube_path, '/', '.png', 2)
                    renderer.UseSphericalHarmonicsOff()
                    renderer.SetEnvironmentTexture(cubemap, True)
                    renderer.UseImageBasedLightingOn()



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
                    if add_colorbar:
                        cb = colorbar(actor, interactor=renderWindowInteractor)
                        # renderer.AddActor(cb)

            elif isinstance(actors, dict):
                print('Dict of actors supplied, adding Dict')
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
                            if add_colorbar:
                                cb = colorbar(sub_actor, interactor=renderWindowInteractor)
                                # renderer.AddActor(cb)
                    elif type(actor) is list:
                        for sub_actor in actor:
                            if use_PBR:
                                renderer.AddActor(add_PBR(sub_actor))
                            else:
                                renderer.AddActor(sub_actor)
                            if add_axes:
                                axes = make_axes(sub_actor, renderer)
                                renderer.AddActor(axes)
                            if add_colorbar:
                                cb = colorbar(sub_actor, interactor=renderWindowInteractor)
                                # renderer.AddActor(cb)
                    elif isinstance(actor, dict):
                        for sub_actor in actor.values():
                            if use_PBR:
                                renderer.AddActor(add_PBR(sub_actor))
                            else:
                                renderer.AddActor(sub_actor)
                            if add_axes:
                                axes = make_axes(sub_actor, renderer)
                                renderer.AddActor(axes)
                            if add_colorbar:
                                cb = colorbar(sub_actor, interactor=renderWindowInteractor)
                                # renderer.AddActor(cb)
                    elif isinstance(actor, tuple):
                        for sub_actor in actor:
                            if use_PBR:
                                renderer.AddActor(add_PBR(sub_actor))
                            else:
                                renderer.AddActor(sub_actor)
                            if add_axes:
                                axes = make_axes(sub_actor, renderer)
                                renderer.AddActor(axes)
                            if add_colorbar:
                                cb = colorbar(sub_actor, interactor=renderWindowInteractor)
                                # renderer.AddActor(cb)

                    else:
                        if use_PBR:
                            renderer.AddActor(add_PBR(actor))
                        else:
                            renderer.AddActor(actor)
                        if add_axes:
                            axes = make_axes(actor, renderer)
                            renderer.AddActor(axes)
                        if add_colorbar:
                            cb = colorbar(actor, interactor=renderWindowInteractor)
                            # renderer.AddActor(cb)

        else:
            if use_PBR:
                renderer.AddActor(add_PBR(actors))
            else:
                renderer.AddActor(actors)
            if add_axes:
                axes = make_axes(actors, renderer)
                renderer.AddActor(axes)
            if add_colorbar:
                cb = colorbar(actor)
                renderer.AddActor(cb)


        renderer.ResetCamera()
        render_camera.Azimuth(90)
        render_camera.SetViewUp(0,0,1)

        renderWindowInteractor.SetInteractorStyle(interactor_utils.MyInteractorStyle(renderWindowInteractor, render_camera, renderWindow))

        # current_actors = renderer.GetActors()
        # print(dir(current_actors))

        # print(current_actors.GetItemAsObject(0))
        sliders = add_sliders(renderWindowInteractor, renderer, initial_values=initial_values, secondary_scalar=secondary_scalar)

        config.sliders = sliders
    # cam_orient_manipulator = vtk.vtkCameraOrientationWidget()
    # cam_orient_manipulator.SetParentRenderer(renderer)
    # # Enable the widget.
    # cam_orient_manipulator.On()

    renderWindow.SetWindowName('Nuclear Fruit Bowl: Shape Generator')
    renderWindow.SetSize(window_size)
    renderWindow.Render()

    export_button = add_export_button(renderWindowInteractor, renderer)
    export_button.On()
    reset_button = add_reset_button(renderWindowInteractor, renderer)
    reset_button.On()

    # cube = interactor_utils.add_indicator_cube(renderWindowInteractor)
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

def add_polyhedron(vertices, faces, labels=None, offset=[0, 0, 0], scalars=None, secondary_offset=None, original_actor=None, rotation=None, generate_normals=True, render_flat=False, opacity=1.0, verbose=False, mesh_color='black', color_map='viridis', c_range=None, representation='surface', interpolate_scalars=True, actor_name=None):

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

    # polydata, tcords = generate_texture_coords(None, None, polydata)

    if generate_normals:
        normal_filter = vtk.vtkPolyDataNormals()
        normal_filter.SetInputData(polydata)
        normal_filter.Update()

    # filter = vtk.vtkGeometryFilter()
    # filter.SetInputData(polydata)
    # filter.MergingOn()
    # filter.Update()
    # polydata = filter.GetOutput()



    # tangents = vtk.vtkPolyDataTangents()
    # tangents.SetComputePointTangents(True)
    # tangents.SetComputeCellTangents(True)
    # if generate_normals:
    #     tangents.SetInputData(normal_filter.GetOutput())
    # else:
    #     tangents.SetInputData(polydata)
    # tangents.Update()


    if original_actor is None:

        # print(normal_filter.GetOutput(), scalars.shape)
        # Create a mapper and actor
        mapper = vtk.vtkDataSetMapper()
        if generate_normals:
            mapper.SetInputData(normal_filter.GetOutput())
        else:
            mapper.SetInputData(polydata)
        # mapper.SetInputData(tangents.GetOutput())

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
            if generate_normals:   
                for val in scalars:
                    col = [0., 0., 0.]
                    lut.GetColor(val, col)
                    col = [int(c * 255) for c in col]
                    cur_color_data.InsertNextTuple3(col[0], col[1], col[2])

            actor.GetMapper().GetInput().GetPointData().SetScalars(cur_color_data)
            actor.GetMapper().SetInterpolateScalarsBeforeMapping(interpolate_scalars)

        actor.GetProperty().SetOpacity(opacity)

        if actor_name is None:
            
            random_string = np.random.randint(10000,9000000)

            if representation == 'wireframe':
                actor_name = 'wireframe-%i' % random_string
            else:
                actor_name = 'surface-%i' % random_string

        if representation == 'wireframe':

            actor.GetProperty().SetRepresentationToWireframe()

        if render_flat:
            actor.GetProperty().SetInterpolationToFlat()
        else:
            actor.GetProperty().SetInterpolationToGouraud()

       
        actor.SetObjectName(actor_name)

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

            if generate_normals:   
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


def add_spherical_function(function_values, secondary_scalars=None, radius=1, scale_mesh=True, add_gridlines=False, function_name=None, colormap='jet', mesh_color='black', absolute_displacement=True, offset=[0,0,0], line_width=2, original_actor=None, opacity=1):

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

            if y_num == y_granularity - 2: # for the endseam of the sphere to join it back up
                tri_1 = [coordination[x_num, y_num], coordination[x_num + 1, y_num], coordination[x_num + 1, 0]]
                triangles.append(tri_1)
                tri_2 = [coordination[x_num, y_num], coordination[x_num + 1, 0], coordination[x_num , 0]]
                triangles.append(tri_2)

            
            elif x_num == x_granularity - 2: # for the endseam of the sphere to join it back up
                # tri_1 = [coordination[x_num, y_num], coordination[-1, y_num], coordination[-1, y_num + 1]]
                # triangles.append(tri_1)
                tri_2 = [coordination[x_num, y_num], coordination[x_num + 1,  y_num + 1], coordination[x_num ,  y_num +1]]
                triangles.append(tri_2)
            else:
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
        surface = add_polyhedron(locs, triangles, scalars=flat_function_values, opacity=opacity, color_map=colormap, offset=offset, original_actor=original_actor)
    else:
        if np.ndim(secondary_scalars) > 1:
            secondary_scalars = secondary_scalars.flatten()
        surface = add_polyhedron(locs, triangles, scalars=secondary_scalars, opacity=opacity, color_map=colormap, offset=offset, original_actor=original_actor)

    actor_dict = dict()

    if function_name is None:
        function_name = 'surface-%i' % np.random.randint(0, 10000000)

    actor_dict.update({function_name : surface})

    if add_gridlines:
        print(quads.shape)
        grid = add_polyhedron(locs, quads, scalars=None, opacity=1.0, color_map=colormap, representation='wireframe', mesh_color=mesh_color, offset=offset)
        grid.GetProperty().SetLineWidth(line_width)
        grid.GetProperty().LightingOff()
        grid.GetProperty().SetRenderLinesAsTubes(True)

        if function_name is None:
            gridline_name = 'mesh lines %i' % np.random.randint(0, 10000000)
        else:
            gridline_name = 'gridlines-%s' % function_name 
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