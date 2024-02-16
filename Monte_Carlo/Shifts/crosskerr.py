import numpy as np
import matplotlib.pyplot as plt

xcList = []
xs1List = []
xs2List = []

for delta in np.arange(-200.0, 201.0, 15.0):
    D1 = 0.5 + delta * 10**-3
    D2 = 1.0
    f = np.sqrt(5.7 / (16 * 0.045))
    g1 = f * 0.09
    g2 = f * 0.075
    chi = -0.045 / 2  # GHz

    M = np.array([[D1, g1, g2], [g1, 0, 0], [g2, 0, D2]])
    eM, vM = np.linalg.eigh(M)

    vM1 = vM[:, 0]
    vM2 = vM[:, 1]
    vM3 = vM[:, 2]

    Norm1 = np.abs(np.vdot(vM1, vM1))
    Norm2 = np.abs(np.vdot(vM2, vM2))
    Norm3 = np.abs(np.vdot(vM3, vM3))

    vnM1 = vM1 / np.sqrt(Norm1)
    vnM2 = vM2 / np.sqrt(Norm2)
    vnM3 = vM3 / np.sqrt(Norm3)

    U = [vnM1, vnM2, vnM3]
    V = np.transpose(U)



    def X(i, j):
        return 2 * chi * np.abs(V[1,i]) ** 2 * np.abs(V[1,j]) ** 2

    xs1List.append([delta, 10**6 * X(1, 1)])
    xs2List.append([delta, 10**6 * X(2, 2)])
    xcList.append([delta, 10**6 * 2 * X(1, 2)])


xs1List = np.array(xs1List)
xs2List = np.array(xs2List)
xcList = np.array(xcList)

plt.figure()
plt.plot(xs1List[:, 0], xs1List[:, 1], 'r', label='xs1')
plt.plot(xs2List[:, 0], xs2List[:, 1], 'g', label='xs2')
plt.plot(xcList[:, 0], xcList[:, 1], 'b--', label='xc')
plt.xlabel('delta [MHz]')
plt.ylabel('Chi [Khz]')
plt.ylim(-1000, 0)
plt.legend()
plt.show()





