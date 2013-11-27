import numpy as np
from dipy.viz import fvtk
from dipy.sims.voxel import multi_tensor, multi_tensor_odf, all_tensor_evecs
from dipy.reconst.shore_cart import ShoreCartModel
from dipy.data import get_data, get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table


def sim_tensor_2x(gtab, angle=90, sphere=None, S0=1., snr=None):

    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    data, sticks = multi_tensor(gtab, mevals, S0,
                                angles=[(-angle / 2., 0), (angle / 2., 0)],
                                fractions=[50, 50], snr=snr)

    mevecs = [all_tensor_evecs(sticks[0]).T,
              all_tensor_evecs(sticks[1]).T]

    odf_gt = multi_tensor_odf(sphere.vertices, [0.5, 0.5], mevals, mevecs)

    return data, sticks, odf_gt


SNR = 15

zeta = 700.
mu = 1/ (2 * np.pi * np.sqrt(zeta))
lambd = 0

fbvals, fbvecs = get_data('3shells_data')
bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
gtab = gradient_table(bvals, bvecs)

sphere = get_sphere('symmetric724').subdivide(1)

radial_orders = [0, 2, 4, 6]
angles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

odfs = np.zeros((len(radial_orders) + 1, len(angles),
                sphere.vertices.shape[0]))


sim_data = np.zeros((len(angles), gtab.bvals.shape[0]))
for (j, angle) in enumerate(angles):
    data, dirs_gt, odf_gt = sim_tensor_2x(gtab, angle, sphere, snr=SNR)
    odfs[0, j] = odf_gt
    sim_data[j] = data

scms = []

for (i, radial_order) in enumerate(radial_orders):
    scm = ShoreCartModel(gtab, radial_order, mu=mu, lambd=lambd)
    scms.append(scm)

smoments = [0, 6]
lambdas = [0, 0.001, 0.01, 0.1, 1, 2]

for smoment in smoments:

    for (k, lambd) in enumerate(lambdas):

        for (i, radial_order) in enumerate(radial_orders):
            scm = scms[i]
            scm.lambd = lambd
            for (j, angle) in enumerate(angles):
                print(radial_order, angle)
                odfs[i + 1, j] = scm.fit(sim_data[j]).odf(sphere, smoment=smoment)

        odfs = odfs[:, None, :]

        ren = fvtk.ren()
        fvtk.add(ren, fvtk.sphere_funcs(odfs, sphere))

        fvtk.camera(ren, [0, -5, 0], [0, 0, 0], viewup=[-1, 0, 0])

        #fvtk.show(ren)
        fname = 'shore_cart_odfs_snr_' + str(SNR) + '_s_' + str(smoment) + '_' + str(k) + '_l_' + str(lambd) + '.png'

        fvtk.record(ren, n_frames=1, out_path=fname, size=(1000, 1000))

        odfs = np.squeeze(odfs)

