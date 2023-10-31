import numpy as np
import matplotlib
import vtk
import config

def make_colormap(c_range=(0,1), color='viridis', cnum=100, invert=False, verbose=False):


# Possible values are: 

# Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r,
# BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu,
# GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r,
# PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r,
# PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r,
# Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu,
# RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r,
# Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu,
# YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r,
# autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r,
# cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r,
# cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray,
# gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow,
# gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot,
# gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r,
# inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral,
# nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r,
# prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spring,
# spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b,
# tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r,
# twilight_shifted, twilight_shifted_r, viridis, viridis_r, winter, winter_r

    if verbose:
        print('Using Matplotlib color tables')
    mpl_data = matplotlib.cm.get_cmap(color)
    cm_data = np.zeros([cnum, 3])
    for i in range(cnum):
        cm_data[i,:] = mpl_data(i/cnum)[0:3]

    colormap = np.asarray(cm_data)

    if invert:
        if verbose:
            print('Using inverted direction')
        colormap = np.flip(colormap, axis=0)

    scale = np.linspace(c_range[0], c_range[1], num=colormap.shape[0])

    colormap = np.vstack([scale, colormap.T]).T

    if verbose:
        print('Colormap made')

    return colormap


def make_LUT(colormap='viridis', invert=False, verbose=False, c_range=(0,1), nan_color=(1,0,0,1), scale_type='linear'):

    colormap = make_colormap(c_range=c_range, color=colormap, invert=invert, verbose=verbose)

    if config.dark_mode:
        colormap = colormap * [1, 0.1, 0.1, 0.1]

    colorSeries = vtk.vtkColorSeries()
    colorSeries.SetNumberOfColors(colormap.shape[0])
    try:
        for cnum, color in enumerate(colormap):
            color = (255*color[1:4]).astype(int) 
            vcolor = vtk.vtkColor3ub(color)
            colorSeries.SetColor(cnum, vcolor)
        lut = vtk.vtkLookupTable()
        colorSeries.BuildLookupTable(lut, colorSeries.ORDINAL)
        lut.SetNanColor(nan_color)
    except TypeError:
        for cnum, color in enumerate(colormap):
            color = (255*color[1:4]).astype(int) 
            vcolor = vtk.vtkColor3ub(color[0], color[1], color[2])
            colorSeries.SetColor(cnum, vcolor)
        lut = vtk.vtkLookupTable()
        colorSeries.BuildLookupTable(lut, colorSeries.ORDINAL)
        lut.SetNanColor(nan_color)
        lut.SetRange(c_range)

    if scale_type == 'linear':

        lut.SetScaleToLinear()

    elif scale_type == 'logarithmic':

        lut.SetScaleToLog10()

    return lut
