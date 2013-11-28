import numpy as np 
from dipy.viz import fvtk
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from dipy.sims.voxel import multi_tensor, multi_tensor_odf, all_tensor_evecs
from dipy.reconst.shore_cart import ShoreCartModel, shore_e0, shore_evaluate_E
from dipy.data import get_data, get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table



def sim_tensor_2x(gtab, angle=90, sphere=None, S0=1., snr=None):

    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    data, sticks = multi_tensor(gtab, mevals, S0,
                                angles=[(90 , 0), (90, angle)],
                                fractions=[50, 50], snr=snr)

    mevecs = [all_tensor_evecs(sticks[0]).T,
              all_tensor_evecs(sticks[1]).T]

    odf_gt = multi_tensor_odf(sphere.vertices, [0.5, 0.5], mevals, mevecs)

    return data, sticks, odf_gt


def fill_qspace_plane(radial_order, coeff, gtab, mu, npoints, angle, sphere, snr):

    q = 1.1 * np.sqrt(gtab.bvals)
    q.max()   

    x = np.arange(-q.max(), q.max(), q.max()/npoints)
    y = np.arange(-q.max(), q.max(), q.max()/npoints)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    Xl = X.ravel()
    Yl = Y.ravel()
    Zl = Z.ravel()

    qlist = np.vstack((Xl,Yl,Zl)).T
    
    bval_draw = Zl ** 2 + Xl ** 2 + Yl ** 2
    bvec_draw = qlist / np.sqrt(bval_draw)[:,None]

    El = shore_evaluate_E(radial_order, coeff, qlist, mu)
    
    Egrid = El.reshape(X.shape)
    gtab_draw = gradient_table(bval_draw , bvec_draw)

    E_gt, _, _ = sim_tensor_2x(gtab_draw, angle=angle, sphere=sphere, snr=snr)

    return X, Y, Egrid, E_gt.reshape(X.shape)



zeta = 700.
mu = 1/ (2 * np.pi * np.sqrt(zeta))
lambd = 0.001

radial_order = 6
angle = 90
fsamples = 'samples.txt'
snr=100
scheme = np.loadtxt(fsamples)

scheme[:, 0] *= 1000

bvals = scheme[:, 0]
bvecs = scheme[:, 1:]

gtab = gradient_table(bvals, bvecs)

sphere = get_sphere('symmetric724')

data, sticks, odf_gt = sim_tensor_2x(gtab, angle=angle, sphere=sphere, snr=snr)

shore_model = ShoreCartModel(gtab, radial_order, mu=mu, lambd=lambd)

shore_fit = shore_model.fit(data)

M = shore_model.cache_get('shore_phi_matrix', key=shore_model.gtab)

coeff = shore_fit.shore_coeff
data_fitted = np.dot(M, coeff)

X, Y, Egrid, E_gt = fill_qspace_plane(radial_order, coeff, gtab, mu, 15, angle, sphere, snr)

fig = plt.figure(1)
ax1 = fig.add_subplot(2, 2, 1, title='data vs fitted signal')
ax1.plot(data.ravel())
ax1.plot(data_fitted.ravel())


print("E0 %f" % shore_e0(radial_order , shore_fit.shore_coeff))

ax1 = fig.add_subplot(2, 2, 4, title='fitted')
ax1.contour(X, Y, Egrid, [.2, .3, .4, .5, .6, .7, .8])
ax1.axis('equal')

ax2 = fig.add_subplot(2, 2, 2, title='ground truth')
ax2.contour(X, Y, E_gt, [.2, .3, .4, .5, .6, .7, .8])
ax2.axis('equal')

plt.show()