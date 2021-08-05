import numpy as np
import matplotlib.pyplot as plt

augment_method = "Specmix"

d = np.load("result_{}/curve.npz".format(augment_method))
loss = d['loss']
acc = d['acc']
best_acc = d['best_acc']

print(best_acc)

plt.plot(loss)
plt.show()
plt.clf()

plt.plot(acc)
plt.show()
plt.clf()