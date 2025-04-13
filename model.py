import numpy as np
import bisect
import matplotlib.pyplot as plt
import matplotlib.ticker as tic

burstAltitude = 86000
launchAltitude = 0

g0 = 9.80665
R = 1.380622 * 6.022169
M0 = 28.9644
P0 = 101.325 #kPa
r0 = 6356766 #meters

segments = [0, 11000, 20000, 32000, 47000, 51000, 71000, 84852] #geopotentialAltitude
lapseRates = [-6.5, 0, 1, 2.8, 0, -2.8, -2, 0] #K/km

def calculateGraviationalAcceleration(geometricAltitude): #meters
    graviationalAcceleration = g0 * (r0 / (r0 + geometricAltitude)) ** 2
    return graviationalAcceleration

def calculateGeopotentialAltitude(geometricAltitude): #meters
    geopotentialAltitude = geometricAltitude * r0 / (geometricAltitude + r0)
    return geopotentialAltitude

baseTemperatures = [288.15] #K

def calculateTemperature(index, geopotentialAltitude): #int, meters
    temperature = baseTemperatures[i] + lapseRates[i] * (geopotentialAltitude - segments[i]) / 1000
    return temperature

for i in range(len(segments) - 1):
    baseTemperatures.append(calculateTemperature(i, segments[i + 1]))

basePressures = [P0]

def calculatePressure(index, geopotentialAltitude): #int, meters
    if lapseRates[i] != 0:
        pressure = basePressures[i] * (baseTemperatures[i] / ((baseTemperatures[i] + lapseRates[i] * (geopotentialAltitude - segments[i]) / 1000))) ** (g0 * M0 / (R * lapseRates[i]))
    else:
        pressure = basePressures[i] * np.exp((-g0 * M0 * (geopotentialAltitude - segments[i]) / 1000) / (R * baseTemperatures[i]))
    return pressure

for i in range(len(segments) - 1):
    basePressures.append(calculatePressure(i, segments[i + 1]))

def calculateDensity(index, pressure, temperature):
    density = pressure * M0 / (R * temperature)
    return density

pressures = []
temperatures = []
densities = []
altitudes = [launchAltitude]
step = 0

while altitudes[-1] <= burstAltitude:
    i = bisect.bisect(segments, altitudes[-1]) - 1
    pressures.append(calculatePressure(i, calculateGeopotentialAltitude(altitudes[-1])))
    temperatures.append(calculateTemperature(i, calculateGeopotentialAltitude(altitudes[-1])))
    densities.append(calculateDensity(i, pressures[-1], temperatures[-1]))
    altitudes.append(altitudes[-1] + 1000)

fig, ax1 = plt.subplots()

ax1.plot(pressures, np.array(altitudes[:-1]) / 1000, "b-", label="Pressure")
ax1.set_xlabel("Pressure (kPa)")
ax1.set_ylabel("Altitude (m)")
ax1.set_xscale('log')
ax1.yaxis.set_major_locator(tic.MultipleLocator(10))
ax1.tick_params(axis = "x", colors = "blue")
ax1.legend(loc='upper right')

ax2 = ax1.twiny()
ax2.plot(densities, np.array(altitudes[:-1]) / 1000, "r--", label="Density")
ax2.set_xlabel("Density (kg/m^3)")
ax2.set_xscale('log')
ax2.tick_params(axis = "x", colors = "red")
ax2.legend(loc='lower left')

ax1.set_xlim(min(min(pressures), min(densities)) / 10 ** 0.5, max(max(pressures), max(densities)) * 10 ** 0.5)
ax1.set_ylim(min(np.array(altitudes[:-1]) / 1000), np.ceil(max(np.array(altitudes[:-1]) / 1000) / 10) * 10)
ax2.set_xlim(ax1.get_xlim())

plt.title("ISA Profile")
plt.grid(True)
plt.tight_layout()
plt.show()