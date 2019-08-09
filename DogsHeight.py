import numpy as rd
import matplotlib.pyplot as plt

labradors = 500
greyhounds = 500

lab_height = 24 + 4 * rd.random.randn(labradors)
gry_height = 28 + 4 * rd.random.randn(greyhounds)
#print(lab_height)
#print(gry_height)

plt.hist([gry_height, lab_height ],stacked=True,color=(['r','g']))
plt.show()