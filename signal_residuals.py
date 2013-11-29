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

def fill_qspace_line(radial_order, coeff, gtab, mu, npoints, angle, sphere, snr):

    q = 2 * np.sqrt(gtab.bvals)
    q.max()   

    x = np.arange(0, q.max(), q.max()/npoints)
    y = np.zeros(x.shape)
    z = np.zeros(x.shape)
    
    qlist = np.vstack((x,y,z)).T
    
    bval_draw = x ** 2 + y ** 2 + z ** 2
    bvec_draw = qlist / (np.sqrt(bval_draw)[:,None]+.0000000001)

    E_line_ft = shore_evaluate_E(radial_order, coeff, qlist, mu)
    
    gtab_draw = gradient_table(bval_draw , bvec_draw)

    E_line_noise, _, _ = sim_tensor_2x(gtab_draw, angle=angle, sphere=sphere, snr=snr)
    E_line_gt, _, _ = sim_tensor_2x(gtab_draw, angle=angle, sphere=sphere, snr=None)

    return x, bval_draw, E_line_ft, E_line_gt, E_line_noise 


zeta = 700.
mu = 1/ (2 * np.pi * np.sqrt(zeta))
lambd = 0.001

radial_order = 8
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

x, b, E_line_ft, E_line_gt, E_line_noise = fill_qspace_line(radial_order, coeff, gtab, mu, 50, angle, sphere, snr)

X, Y, Egrid, E_gt = fill_qspace_plane(radial_order, coeff, gtab, mu, 15, angle, sphere, snr)

fig = plt.figure(1)
#fig.title('E(0) = {:3.3f}'.format(shore_e0(radial_order , shore_fit.shore_coeff)))
ax1 = fig.add_subplot(2, 2, 1, title='data vs fitted signal')
ax1.plot(data.ravel())
ax1.plot(data_fitted.ravel())



ax3 = fig.add_subplot(2, 2, 3, title='radial data vs fitted signal')
ax3.plot(x, E_line_gt)
ax3.plot(x, E_line_ft)
ax3.plot(x, E_line_noise,'r.')
ax3.plot(sqrt([5000,5000]), [1.2,0.001],'.k--')
ax3.plot(sqrt([4000,4000]), [1.2,0.001],'.k--')
ax3.plot(sqrt([3000,3000]), [1.2,0.001],'.k--')
ax3.plot(sqrt([2000,2000]), [1.2,0.001],'.k--')
ax3.plot(sqrt([1000,1000]), [1.2,0.001],'.k--')


ax4 = fig.add_subplot(2, 2, 4, title='radial data vs fitted signal')
ax4.semilogy(x, E_line_gt)
ax4.semilogy(x, E_line_ft)
ax4.semilogy(x, E_line_noise,'r.')
ax4.semilogy(sqrt([5000,5000]), [1.2,0.001],'.k--')
ax4.semilogy(sqrt([4000,4000]), [1.2,0.001],'.k--')
ax4.semilogy(sqrt([3000,3000]), [1.2,0.001],'.k--')
ax4.semilogy(sqrt([2000,2000]), [1.2,0.001],'.k--')
ax4.semilogy(sqrt([1000,1000]), [1.2,0.001],'.k--')


print("E0 %f" % shore_e0(radial_order , shore_fit.shore_coeff))

#ax4 = fig.add_subplot(2, 2, 4, title='fitted')
#ax4.contour(X, Y, Egrid, [.2, .3, .4, .5, .6, .7, .8])
#ax4.axis('equal')

print(E_line_ft[0])

ax2 = fig.add_subplot(2, 2, 2, title='data')
ax2.contour(X, Y, Egrid, [.2, .3, .4, .5, .6, .7, .8])
ax2.contour(X, Y, E_gt, [.2, .3, .4, .5, .6, .7, .8])
ax2.axis('equal')

plt.show()