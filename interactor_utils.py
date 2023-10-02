import vtk
import numpy as np
import math

# class MyInteractorStyle(vtk.vtkInteractorStyleRubberBand3D):
class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """[VTK class definitions to allow for more natural camera movement (trackpad style)
        and keybindings for extra functionality (camera movement, printscreens etc.)]

    Arguments:
      expects: renderWindowInteractor, render_camera, renderWindow
    """

    def __init__(self, parent, camera, renderer):
        self.parent = parent
        self.camera = camera
        self.renderer = renderer

        self.verbose = False
        self.auto_up = True
        self.AddObserver("MiddleButtonPressEvent", self.middle_button_press_event)
        self.AddObserver("MiddleButtonReleaseEvent", self.middle_button_release_event)
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release_event)
        self.AddObserver("KeyPressEvent", self.keyPressEvent)
        # self.AddObserver('AnnotationChangedEvent', selectionCallback)
        # self.AutoAdjustCameraClippingRange(True)

    def camera_zoom_in(self, step=2):

        old = np.asarray(self.camera.GetPosition())

        new = old *0.9

        if self.auto_up:
            up = np.asarray([0, 0, 1])
            self.camera.SetViewUp(up)

        self.camera.SetPosition(tuple(new))

        self.renderer.Render()
        return


    def camera_zoom_out(self, step=2):

        old = np.asarray(self.camera.GetPosition())

        new = old * 1.1

        if self.auto_up:
            up = np.asarray([0, 0, 1])
            self.camera.SetViewUp(up)

        self.camera.SetPosition(tuple(new))
        
        self.renderer.Render()
        return



    def rotate_clockwise(self, step=2):

        old = list(self.camera.GetPosition())

        x_coord = old[0]
        y_coord = old[1]

        if x_coord >= 0:
            if y_coord >= 0:
                x_sign = -1
                y_sign = 1
            else:
                x_sign = 1
                y_sign = 1
        elif y_coord >= 0:
            x_sign = -1
            y_sign = -1
        else:
            x_sign = 1
            y_sign = -1

        scale = 0.05

        del_x = np.abs(y_coord) * scale * x_sign
        del_y = np.abs(x_coord) * scale * y_sign
        hypot_1 = math.hypot(old[0], old[1])

        new_x =  old[0] + del_x
        new_y =  old[1] + del_y

        hypot_2 = math.hypot(new_x, new_y)

        rescale = hypot_1/hypot_2

        new = old
        new[0] = (new_x * rescale)
        new[1] = (new_y * rescale)

        if self.auto_up:
            up = np.asarray([0, 0, 1])
            self.camera.SetViewUp(up)

        self.camera.SetPosition(tuple(new))

        self.renderer.Render()
        return


    def rotate_anticlockwise(self):


        old = list(self.camera.GetPosition())

        x_coord = old[0]
        y_coord = old[1]

        if x_coord > 0:
            if y_coord > 0:
                x_sign = 1
                y_sign = -1
            else:
                x_sign = -1
                y_sign = -1
        elif y_coord > 0:
            x_sign = 1
            y_sign = 1
        else:
            x_sign = -1
            y_sign = 1

        scale = 0.05

        del_x = np.abs(y_coord) * scale * x_sign
        del_y = np.abs(x_coord) * scale * y_sign
        hypot_1 = math.hypot(old[0], old[1])

        new_x =  old[0] + del_x
        new_y =  old[1] + del_y

        hypot_2 = math.hypot(new_x, new_y)

        rescale = hypot_1/hypot_2

        new = old
        new[0] = (new_x * rescale)
        new[1] = (new_y * rescale)

        if self.auto_up:
            up = np.asarray([0, 0, 1])
            self.camera.SetViewUp(up)

        self.camera.SetPosition(tuple(new))

        self.renderer.Render()
        return


    def rotate_upclockwise(self):


        old = list(self.camera.GetPosition())

        x_coord = old[0]
        y_coord = old[1]

        # print(x_coord, y_coord)

        d_coord = math.hypot(x_coord, y_coord)

        z_coord = old[2]

        if z_coord >= 0:
           z_sign = 1
           d_sign = -1
        else:
            z_sign = 1
            d_sign = 1

        scale = 0.05

        del_z = np.abs(d_coord) * scale * z_sign
        del_d = np.abs(z_coord) * scale * d_sign
        hypot_1 = math.hypot(d_coord, z_coord)

        new_z =  z_coord + del_z
        new_d =  d_coord + del_d

        hypot_2 = math.hypot(new_z, new_d)
        rescale = hypot_1/hypot_2

        new = old
        new[2] = (new_z * rescale)
        new_d = (new_d * rescale)

        rescale_2 = new_d/d_coord

        new[0] = (x_coord * rescale_2)
        new[1] = (y_coord * rescale_2)

        if self.auto_up:
            up = np.asarray([0, 0, 1])
            self.camera.SetViewUp(up)

        self.camera.SetPosition(tuple(new))

        self.renderer.Render()
        return


    def rotate_downclockwise(self):

        old = list(self.camera.GetPosition())

        x_coord = old[0]
        y_coord = old[1]

        d_coord = math.hypot(x_coord, y_coord)

        z_coord = old[2]

        if z_coord >= 0:
           z_sign = -1
           d_sign = +1
        else:
            z_sign = -1
            d_sign = -1

        scale = 0.05

        del_z = np.abs(d_coord) * scale * z_sign
        del_d = np.abs(z_coord) * scale * d_sign
        hypot_1 = math.hypot(d_coord, z_coord)

        new_z =  z_coord + del_z
        new_d =  d_coord + del_d

        hypot_2 = math.hypot(new_z, new_d)
        rescale = hypot_1/hypot_2

        new = old
        new[2] = (new_z * rescale)
        new_d = (new_d * rescale)

        rescale_2 = new_d/d_coord

        new[0] = (x_coord * rescale_2)
        new[1] = (y_coord * rescale_2)

        if self.auto_up:
            up = np.asarray([0, 0, 1])
            self.camera.SetViewUp(up)

        self.camera.SetPosition(tuple(new))

        self.renderer.Render()
        return


    def screenshot(self):

        print(self.camera)
        print(self.camera.GetRoll())
        print(self.camera.Get)

        w2if = vtk.vtkWindowToImageFilter()
        # print(dir(w2if))
        w2if.SetScale(4)
        w2if.SetInput(self.renderer)
        w2if.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName("screenshot.png")
        writer.SetInputData(w2if.GetOutput())
        writer.Write()

    def keyPressEvent(self, obj, event):
        key = str(self.parent.GetKeySym())
        # check here for available keypresses supported by Qt & VTK
        # https://github.com/Kitware/VTK/blob/master/GUISupport/Qt/QVTKInteractorAdapter.cxx
        if key == 'Left':
            self.rotate_clockwise()
        if key == 'Right':
            self.rotate_anticlockwise()
        if key == 'Up':
            self.rotate_upclockwise()
        if key == 'Down':
            self.rotate_downclockwise()
        if key == 'c':
            self.screenshot()
        if key == 'equal':
            self.camera_zoom_in()
        if key == 'minus':
            self.camera_zoom_out()
        if key == 'o':
            self.switch_rendering_mode()
        return


    def left_button_press_event(self, obj, event):
        if self.verbose:
            print("left Button pressed")
        self.OnLeftButtonDown()
        return


    def left_button_release_event(self, obj, event):
        if self.verbose:
            print("left Button released")
        self.OnLeftButtonUp()
        return


    def middle_button_press_event(self, obj, event):
        if self.verbose:
            print("Middle Button pressed")
        self.OnMiddleButtonDown()
        return


    def middle_button_release_event(self, obj, event):
        if self.verbose:
            print("Middle Button released")
        self.OnMiddleButtonUp()
        return

    def switch_rendering_mode(self):
        # print(dir(self.camera))
        projection_mode = self.camera.GetParallelProjection()
        if projection_mode == 1:
            self.camera.SetParallelProjection(False)
            projection_mode = self.camera.GetParallelProjection()
            # print(projection_mode)
            print('Camera set to Perspective Projection')
        else:
            self.camera.SetParallelProjection(True)
            projection_mode = self.camera.GetParallelProjection()
            # print(projection_mode)
            print('Camera set to Parallel Projection')

        self.renderer.Render()


def add_indicator_cube(interactor):

    colors = vtk.vtkNamedColors()

    axes_actor = vtk.vtkAnnotatedCubeActor()
    axes_actor.SetXPlusFaceText('L')
    axes_actor.SetXMinusFaceText('R')
    axes_actor.SetYMinusFaceText('I')
    axes_actor.SetYPlusFaceText('S')
    axes_actor.SetZMinusFaceText('P')
    axes_actor.SetZPlusFaceText('A')
    axes_actor.GetTextEdgesProperty().SetColor(colors.GetColor3d("White"))
    axes_actor.GetTextEdgesProperty().SetLineWidth(2)
    axes_actor.GetCubeProperty().SetColor(colors.GetColor3d("Blue"))
    marker = vtk.vtkOrientationMarkerWidget()
    marker.SetOrientationMarker(axes_actor)
    marker.SetInteractor(interactor)
    marker.EnabledOn()
    marker.InteractiveOn()
    marker.SetViewport(0.9, 0.0, 1.0, 0.1)

    return marker

def add_camera_widget(renderer):
    cam_orient_manipulator = vtk.vtkCameraOrientationWidget()
    cam_orient_manipulator.SetParentRenderer(renderer)
    # Enable the widget.
    cam_orient_manipulator.On()
    return cam_orient_manipulator

def set_passes(renderer, use_ssao=False):



    lightsP = vtk.vtkLightsPass()
    opaqueP = vtk.vtkOpaquePass()
    translucentP = vtk.vtkTranslucentPass()
    volumeP = vtk.vtkVolumetricPass()

    collection = vtk.vtkRenderPassCollection()
    overlayP = vtk.vtkOverlayPass()

    # opaque passes
    if use_ssao:

        bounds = np.asarray(renderer.ComputeVisiblePropBounds())

        b_r = np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])

        occlusion_radius = b_r * 0.1
        occlusion_bias = b_r * 0.001

        ssaoCamP = vtk.vtkCameraPass()
        ssaoCamP.SetDelegatePass(opaqueP)
        ssaoP = vtk.vtkSSAOPass()
        ssaoP.SetRadius(occlusion_radius)
        ssaoP.SetDelegatePass(ssaoCamP)
        ssaoP.SetBias(occlusion_bias)
        ssaoP.SetBlur(True)
        ssaoP.SetKernelSize(256)
        
        collection.AddItem(ssaoCamP)
        collection.AddItem(ssaoP)



    collection.AddItem(overlayP)
    collection.AddItem(lightsP)
    collection.AddItem(opaqueP)




    # translucent and volumic passes
    ddpP = vtk.vtkDualDepthPeelingPass()
    ddpP.SetTranslucentPass(translucentP)
    ddpP.SetVolumetricPass(volumeP)
    collection.AddItem(ddpP)

    sequence = vtk.vtkSequencePass()
    sequence.SetPasses(collection)

    fxaaP = vtk.vtkOpenGLFXAAPass()
    fxaaP.SetDelegatePass(sequence)

    camP = vtk.vtkCameraPass()
    camP.SetDelegatePass(fxaaP)


    # overlayP.SetDelegatePass(fxaaP)

    renderer.SetPass(camP)

    return renderer
