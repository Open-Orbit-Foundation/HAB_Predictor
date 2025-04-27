import numpy as np
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
from balloon import Balloon
from model import Model

dataSets = []

balloons = [
    Balloon(0.3 + 0.6, 1.2, 19.8 * 0.3048, 0.55, 1.2, math.pi / 4 * (2 * 0.3048) ** 2),
    Balloon(0.6 + 0.6, 1.2, 19.8 * 0.3048, 0.55, 1.2, math.pi / 4 * (2 * 0.3048) ** 2),
    Balloon(0.8 + 0.6, 1.6, 19.8 * 0.3048, 0.55, 1.2, math.pi / 4 * (2 * 0.3048) ** 2)
]

launchAltitude = 1400

start_time = time.perf_counter()
Model(balloons, launchAltitude, dataSets).altitudeModel()
end_time = time.perf_counter()
print(f"Runtime: {end_time - start_time}")

fig, ax = plt.subplots()
for i, (xValues, yValues) in enumerate(dataSets):
    ax.plot(xValues, np.array(yValues) / 1000, label = f"Balloon {i + 1}")
ax.set_xlabel("Time (min)")
ax.xaxis.set_major_locator(tic.MultipleLocator(10))
ax.xaxis.set_minor_locator(tic.AutoMinorLocator(2))
ax.set_ylabel("Altitude (km)")
ax.yaxis.set_major_locator(tic.MultipleLocator(5))
ax.yaxis.set_minor_locator(tic.AutoMinorLocator(6))
plt.title("Mission Altitude Profile(s)")
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.show()