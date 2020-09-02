# coding=utf-8
import os
import io
import re
import collections as co

import tables
import numpy as np
import pandas as pd
import pyzdde.zdde as pyz

import PyEchelle


class Spectrograph:
    def __init__(self, blaze, grmm, name):
        self.blaze = blaze
        self.grmm = grmm
        self.name = name


def wavelength_to_xy_config(lad, ind, band='yj'):
    """
    convert wavelength to X-Y position
    Input: wavelength in microns, index (1 less than
    desired configuration number)

    Returns X and Y position on the detector in mm for + and - sides of slit
    """
    if band.lower() == 'yj':
        band_string = ""
    elif band.lower() == "hk":
        band_string = "_HK"
    Xp = np.loadtxt('XY_eq{}/X_plus.dat')  # second order fit parameters for x+
    Xm = np.loadtxt('XY_eq{}/X_minus.dat')
    Yp = np.loadtxt('XY_eq{}/Y_plus.dat')
    Ym = np.loadtxt('XY_eq{}/Y_minus.dat')

    # ind = config - 1 #for 0-based indexing
    # Apply fit parameters for that order
    pxp = np.poly1d(Xp[ind])
    pxm = np.poly1d(Xm[ind])
    pyp = np.poly1d(Yp[ind])
    pym = np.poly1d(Ym[ind])

    Xla = [pxp(lad), pxm(lad)]  # X coordinates of + and- slit
    Yla = [pyp(lad), pym(lad)]  # Y coordinates of + and- slit

    return Xla, Yla


def get_psf_txt_lines(config, wave, band='YJ', use_centroid=False, base_path='.'):
    if use_centroid:
        centroid_prefix = ""
    else:
        centroid_prefix = "no_"
    psf_path = os.path.join(
        base_path, "PSF_{}".format(band), "{}use_centroid".format(centroid_prefix),
        "config_{}_wave_{}.txt".format(config, wave)
    )
    with io.open(psf_path, mode='r', encoding='utf-16') as f:
        psf_txt = f.readlines()
        f.close()
    return psf_txt


def parse_psf_txt_lines(psf_txt_lines):
    def parse_wavelength():
        # 0.9194 µm at 0.0000, 0.0000 (deg).
        return np.double(psf_txt_lines[8].split(" ")[0].strip())

    def parse_data_spacing():
        # Data spacing is 0.422 µm.
        data_line = psf_txt_lines[9]
        split_txt = data_line.split(" is ")[1]
        spacing = split_txt.split(" ")[0]
        return np.double(spacing)

    def parse_pupil_grid():
        # Pupil grid size: 32 by 32
        data_line = psf_txt_lines[12]
        split_txt = data_line.split(": ")[1]
        x, y = split_txt.split(" by ")
        return np.int(x), np.int(y)

    def parse_image_grid():
        # Image grid size: 32 by 32
        data_line = psf_txt_lines[13]
        split_txt = data_line.split(": ")[1]
        x, y = split_txt.split(" by ")
        return np.int(x), np.int(y)

    def parse_center_point():
        # Center point is: 17, 17
        data_line = psf_txt_lines[14]
        split_txt = data_line.split(": ")[1]
        x, y = split_txt.split(", ")
        return np.int(x), np.int(y)

    def parse_center_coords():
        # Center coordinates  :   6.20652974E+00,  -4.18352207E-01 Millimeters
        data_line = psf_txt_lines[15]
        split_txt = data_line.split(": ")[1]
        x, y = split_txt.split(", ")
        x = x.strip()
        y = y.strip()
        y = y.split(" ")[0]
        return np.double(x), np.double(y)

    def parse_data_area():
        # Data area is 13.517 by 13.517 µm.
        data_line = psf_txt_lines[10]
        split_txt = data_line.split("is ")[1]
        x, y = split_txt.split(" by ")
        return np.double(x.strip())

    def parse_headers():
        header = {
            "wavelength": parse_wavelength(),
            "dataSpacing": parse_data_spacing(),
            "pupilGridX": parse_pupil_grid()[0],
            "pupilGridY": parse_pupil_grid()[1],
            "imgGridX": parse_image_grid()[0],
            "imgGridY": parse_image_grid()[1],
            "centerPtX": parse_center_point()[0],
            "centerPtY": parse_center_point()[1],
            "centerCoordX": parse_center_coords()[0],
            "centerCoordY": parse_center_coords()[1],
            "dataArea": parse_data_area(),
        }
        psfi = co.namedtuple(
            'PSFinfo',
            ['dataSpacing', 'dataArea', 'pupilGridX', 'pupilGridY', 'imgGridX', 'imgGridY', 'centerPtX',
             'centerPtY', 'centerCoordX', 'centerCoordY'])
        psfInfo = psfi(
            header['dataSpacing'], header['dataArea'], header['pupilGridX'], header['pupilGridY'], header['imgGridX'],
            header['imgGridY'], header['centerPtX'], header['centerPtY'], header['centerCoordX'], header['centerCoordY']
        )
        return psfInfo

    def parse_data():
        start_line = 22
        end_line = len(psf_txt_lines)
        data_lines = psf_txt_lines[start_line:end_line]
        data_txt = [(txt.strip()).split("\t  ") for txt in data_lines]
        data = [[np.double(intensity) for intensity in line] for line in data_txt]
        # return data
        return np.swapaxes(data, 0, 1)

    return parse_wavelength(), parse_headers(), parse_data()


def get_psfs(conf_orders, wave_range=8, band='YJ'):
    psfs = {}
    for conf_order in range(len(conf_orders)):
        order = conf_orders[conf_order]
        psfs[order] = {}
        for wave in range(wave_range):
            psf_txt = get_psf_txt_lines(conf_order+1, wave+1, band)
            psf = parse_psf_txt_lines(psf_txt)
            psfs[order][psf[0]] = (psf[1], psf[2])
    return psfs


default_config_to_order_array = range(30, 45)
default_config_to_order_array.reverse()
default_config_to_order_array = np.asarray(default_config_to_order_array)
default_ccd = PyEchelle.CCD(2048, 2048, 19, 'y', name='H2RG')


def affine_tsv_filename():
    filenames = os.listdir(".")
    affine_filenames = [re.findall('affine_\d+.tsv', f) for f in filenames]
    found_ints = [re.findall('\d+', f[0]) for f in affine_filenames if f]
    found_ints = [int(i[0]) for i in found_ints if i]
    if found_ints:
        found_int = max(found_ints) + 1
    else:
        found_int = 0
    return "affine_{0:03d}.tsv".format(found_int)


def do_affine_transformation_calculation(
        ccd=default_ccd, config_to_order_array=default_config_to_order_array, band='YJ', sw=80, sh=800
):
    """
    Calculates Affine Matrices that describe spectrograph

    The spectrograph can be described by affine transformations from the input slit to the focal plane.
    an affine transofmration can be described by a 3x3 matrix.
    this function calculates the 3x3 matrix per wavelength and order that matches the input slit to the focal plane

    :param band:
    :type band:
    :param config_to_order_array:
    :type config_to_order_array:
    :param ccd:
    :type ccd: PyEchelle.CCD
    :param fw: fiber/slit width [microns]
    :param fh: fiber/slit height [microns]
    :return:
    """
    from skimage import transform as tf
    ray_trace_csv = 'RIMAS_{}_affine_dependencies.csv'.format(band.upper())
    df = pd.read_csv(ray_trace_csv, encoding='utf-16')
    df['config'] = df['config'].astype(np.int)
    df['order'] = config_to_order_array[df['config']-1]

    unique_orders = df['order'].unique()
    fields = df[['fieldy', 'fieldx']]
    unique_fields = fields.drop_duplicates()
    unique_fields_array = unique_fields.to_numpy()
    nfields = len((unique_fields_array.tolist()))

    # norm_field = np.zeros(fields.shape)
    # norm_field[fields > 0] = 1
    # norm_field[fields < 0] = -1
    norm_field = fields.loc[0:(nfields-1), :]
    norm_field = norm_field.to_numpy()
    norm_field = norm_field.astype(np.int)
    norm_field_list = norm_field.tolist()
    # nw = fields[0].max()
    # nh = fields[1].max()
    fw = sw
    fh = sh
    sampling_input_x = fw
    sampling_input_y = fh
    res = {
        'MatricesPerOrder': np.int(unique_orders.shape[0]),
        'norm_field': norm_field_list,
        'sampling_input_x': np.int(sampling_input_x)
    }

    print('Field width: ' + str(fw))
    print('Field height: ' + str(fh))

    res['field_width'] = np.double(fw)
    res['field_height'] = np.double(fh)
    # sampling_x = sampling_input_x
    # sampling_y = sampling_input_x * fh / fw

    src = np.array(norm_field, dtype=float)
    src[:, 0] -= np.min(src[:, 0])
    src[:, 1] -= np.min(src[:, 1])

    src[:, 0] /= np.max(src[:, 0])
    src[:, 1] /= np.max(src[:, 1])

    # src[:, 0] *= sampling_x
    # src[:, 1] *= sampling_y

    # ppp = []
    dst_x = df['y'].to_numpy()
    dst_y = df['x'].to_numpy()
    orders = df['order'].to_numpy()
    wavelength = df['wavelength'].to_numpy()

        # ppp.append(np.array(self.do_spectral_format(nPerOrder=nPerOrder, FSRonly=False, hx=f[0], hy=f[1])))
    # ppp = np.array(ppp)
    dst_x = np.array(dst_x)
    dst_y = np.array(dst_y)
    dst = np.vstack((dst_x, dst_y))
    dst /= (ccd.pixelSize / 1000.)
    dst += ccd.Nx / 2
    dst = dst.reshape(2, len(dst[0]) / nfields, nfields).transpose((1, 2, 0))

    orders = orders.reshape((len(orders) / nfields, nfields))
    wavelength = wavelength.reshape((len(wavelength) / nfields, nfields))

    affine_matrices = {}
    transformations = {}
    p_headers = ["p{:d}".format(i) for i in range(nfields)]
    src_headers = ["src{:d}".format(i) for i in range(nfields)]
    affine_tsv_headers = ["order", "wavelength"] + p_headers + src_headers + [
        "rotation", "scale0", "scale1", "shear", "translation0", "translation1"
    ]
    affine_save_lines = ["\t".join(affine_tsv_headers)]
    for order, wavel, p in zip(orders, wavelength, dst):
        print("affine transformation inputs {} {}".format(src,p))
        p_list = [i for i in p]
        src_list = [i for i in src]
        inputs_list = [order[0], wavel[0]] + p_list + src_list
        params = tf.estimate_transform('affine', src, p)
        params_list = [
            params.rotation, params.scale[0], params.scale[1],
            params.shear, params.translation[0], params.translation[1]
        ]
        affine_save_line = inputs_list + params_list
        affine_save_lines.append("\t".join([str(i) for i in affine_save_line]))
        if affine_matrices.has_key(order[0]):
            affine_matrices[order[0]].update({wavel[0]: np.array(params_list)})
        else:
            affine_matrices[order[0]] = {wavel[0]: np.array(params_list)}
    with open(affine_tsv_filename(), 'w') as f:
        f.write("\n".join(affine_save_lines))
    res['matrices'] = affine_matrices
    return res


def calculate_blaze_wavelength(blaze_angle_deg, gpmm, order):
    blaze_angle = np.deg2rad(blaze_angle_deg)
    d = 1000 / gpmm
    blaze_wl = 2 * d * np.sin(blaze_angle) / order
    print("order: {}    blaze_wl: {}".format(order, blaze_wl))
    return blaze_wl


def generate_orders(min_order, max_order, blaze_angle_deg, gpmm):
    order_range = range(min_order, max_order+1)
    orders = {}
    for order in order_range:
        fsr_min = 0
        fsr_max = 10
        wavelength_min = 0
        wavelength_max = 10
        blaze_wavelength = calculate_blaze_wavelength(blaze_angle_deg, gpmm, order)
        orders[order] = PyEchelle.Order(order, blaze_wavelength, wavelength_min, wavelength_max, fsr_min, fsr_max)
    return orders


class RIMAS(PyEchelle.Echelle):
    def __init__(self, ln=None, name=''):
        PyEchelle.Echelle.__init__(self, ln, name)

    def get_psfs(self, nPerOrder=1, fieldnumber=3, fieldposition=[0., 0.]):
        get_psfs()


if __name__ == '__main__':
    # filename = 'RIMAS_YJ.hdf'
    # spec = Spectrograph(49.9, 1000/40, 'RIMAS_YJ')
    # PyEchelle.save_spectrograph_info_to_hdf(filename, spec)
    # ccd = PyEchelle.CCD(4096, 4096, 10, dispersionDirection='y')
    # PyEchelle.save_CCD_info_to_hdf(filename, ccd)
    # print(parse_psf_txt_lines(get_psf_txt_lines(1, 1)))

    ln = pyz.createLink()
    filename = os.path.join(
        r'ZEMAX',
        'RIMAS_band1_echelle_capone_130711a_jc-analysis-190316a.zmx')
    ln.zLoadFile(filename)

    spectrograph = PyEchelle.Echelle(ln, 'RIMAS_YJ')
    spectrograph.analyseZemaxFile(echellename='ech s2', blazename='Blaze', gammaname='Gamma')
    spectrograph.minord = 30
    spectrograph.maxord = 44
    spectrograph.blaze = 49.9
    spectrograph.gamma = 0
    spectrograph.grmm = 1000 / 40
    spectrograph.setCCD(PyEchelle.CCD(2048, 2048, 19, name='H2RG'))

    config_array = range(30, 45)
    config_array.reverse()
    config_array = np.asarray(config_array)

    att = do_affine_transformation_calculation(spectrograph.CCD, config_array, 'YJ')
    psfs = get_psfs(config_array, 8, 'YJ')

    directory = r'..\..\Documents\echelle\spectrographs'
    files = os.listdir(directory)
    iterations = [re.findall('\d+', f) for f in files]
    iterations = [int(i[0]) for i in iterations if i != []]
    iteration = max(iterations) + 1
    filename = os.path.join(directory, 'RIMAS_YJ_v{:d}.hdf'.format(iteration))
    PyEchelle.save_spectrograph_info_to_hdf(filename, spectrograph)
    PyEchelle.save_CCD_info_to_hdf(filename, spectrograph.CCD)
    PyEchelle.save_transformation_to_hdf(filename, att, 1)
    PyEchelle.save_psfs_to_hdf(filename, psfs)
