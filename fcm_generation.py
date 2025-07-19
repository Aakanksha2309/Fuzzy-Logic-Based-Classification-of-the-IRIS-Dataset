import numpy as np
import matplotlib.pyplot as plt
from fcmeans import FCM
from graph import simple_fuzzy_plot
def fcm_generation(sat_data, c, img_name, m=2.5, max_iter=100):
    x1, _ = sat_data.shape
    x = sat_data
    options = {'m': m, 'max_iter': max_iter, 'error': 1e-5, 'random_state': 1}
    fcm = FCM(n_clusters=c, m=options['m'], max_iter=options['max_iter'],
              error=options['error'], random_state=options['random_state'])
    fcm.fit(x)
    U = fcm.u
    me = fcm.centers
    print('U.shape:', U.shape)  # Debug print
    I_fcm = np.argmax(U, axis=1) + 1
    I2 = np.uint8(np.round(I_fcm - 1)).reshape(15, 10)
    return me, U

# def fcm_generation(sat_data, c, img_name):
#     x1, _ = sat_data.shape
#     x = sat_data
#     options = {'m': 2.5, 'max_iter': 100, 'error': 1e-5, 'random_state': 1}

#     # FCM clustering
#     fcm = FCM(n_clusters=c, m=options['m'], max_iter=options['max_iter'],
#               error=options['error'], random_state=options['random_state'])
#     fcm.fit(x)
#     U = fcm.u
#     me = fcm.centers

#     # Debug: print U shape
#     print('U.shape:', U.shape)

#     # Get hard cluster labels (axis=1 for samples)
#     I_fcm = np.argmax(U, axis=1) + 1
#     I2 = np.uint8(np.round(I_fcm - 1)).reshape(15, 10)
#     clims = [1, c]

#     plt.figure()
#     plt.imshow(I2, cmap='jet', vmin=clims[0], vmax=clims[1])
#     plt.colorbar()
#     plt.axis('off')
#     plt.savefig(f'results/{img_name}fcm.jpeg', bbox_inches='tight')
#     plt.close()

#     return me, U
