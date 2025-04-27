import numpy as np
import time
import math
import bisect
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
from balloon import Balloon
from atmosphere import standardAtmosphere

start_time = time.perf_counter()

dataSets = []
balloons = [
    Balloon(0.3 + 0.6, 1.2, 19.8 * 0.3048, 0.55, 1.2, math.pi / 4 * (2 * 0.3048) ** 2),
    Balloon(0.6 + 0.6, 1.2, 19.8 * 0.3048, 0.55, 1.2, math.pi / 4 * (2 * 0.3048) ** 2),
    Balloon(0.8 + 0.6, 1.6, 19.8 * 0.3048, 0.55, 1.2, math.pi / 4 * (2 * 0.3048) ** 2)
]

launchAltitude = 1400

atmosphere = standardAtmosphere()

def calculateAscentRate(dragCoefficient, gravity, density, volume, mass, crossSection):
    ascentRate = np.sign(density * volume - mass) * (2 * gravity * abs(density * volume - mass) / (dragCoefficient * density * crossSection)) ** 0.5
    return ascentRate

def calculateDescentRate(dragCoefficient, gravity, density, mass, crossSection):
    descentRate = - ((2 * gravity * mass) / (dragCoefficient * density * crossSection)) ** 0.5
    return descentRate

volumes = []
pressures = []
temperatures = []
densities = []
crossSections = []
ascentRates = []
altitudes = [launchAltitude]
times = [0]
timeStep = 10 #sec

for balloon in balloons:
    volumes.clear()
    pressures.clear()
    temperatures.clear()
    densities.clear()
    crossSections.clear()
    ascentRates.clear()
    altitudes = [launchAltitude]
    times = [0]
    volumes = [balloon.gas]
    
    while volumes[-1] <= balloon.burstVolume():
        i = bisect.bisect(atmosphere.segments, altitudes[-1]) - 1
        pressures.append(atmosphere.Pressure(i, altitudes[-1]))
        temperatures.append(atmosphere.Temperature(i, altitudes[-1]))
        densities.append(atmosphere.Density(pressures[-1], temperatures[-1]))
        volumes.append(balloon.Volume(pressures[-1]))
        crossSections.append(balloon.crossSection(volumes[-1]))
        ascentRates.append(calculateAscentRate(balloon.dragCoefficient, atmosphere._Gravity(altitudes[-1]), densities[-1], volumes[-1], balloon.totalMass(), crossSections[-1]))
        
        if ascentRates[i] > 0:
            altitudes.append(altitudes[-1] + ascentRates[-1] * timeStep)
        else:
            break
        
        times.append(times[-1] + timeStep / 60)

    while altitudes[-1] > altitudes[0]:
        i = bisect.bisect(atmosphere.segments, altitudes[-1]) - 1
        pressures.append(atmosphere.Pressure(i, altitudes[-1]))
        temperatures.append(atmosphere.Temperature(i, altitudes[-1]))
        densities.append(atmosphere.Density(pressures[-1], temperatures[-1]))
        ascentRates.append(calculateDescentRate(balloon.parachuteDragCoefficient, atmosphere._Gravity(altitudes[-1]), densities[-1], balloon.mass, balloon.parachuteCrossSection))
        altitudes.append(altitudes[-1] + ascentRates[-1] * timeStep)
        times.append(times[-1] + timeStep / 60)

    if altitudes[-1] == altitudes[0]:
        del times[-1]
        del altitudes[-1]
    dataSets.append((times[:-1], altitudes[:-1]))

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