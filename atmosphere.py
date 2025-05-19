import numpy as np
import bisect

class standardAtmosphere:
    segments = [0, 11000, 20000, 32000, 47000, 51000, 71000, 84852]
    lapseRates = [-6.5, 0, 1, 2.8, 0, -2.8, -2, 0]
    standardTemperature = 288.15
    standardPressure = 101.325
    gravitationalAcceleration = 9.80665
    gasConstant = 1.380622 * 6.022169
    mixedMolecularWeight = 28.9644
    earthRadius = 6356766
    #altitudes_cache = np.linspace(0, 80000, 801)

    def __init__(self):
        self._base_temps = self._baseTemperatures()
        self._base_pressures = self._basePressures()
        #self.atmospheric_properties = [self._Qualities(alt) for alt in self.altitudes_cache]

    def _Altitude(self, geometricAltitude): #meters
        geopotentialAltitude = geometricAltitude * self.earthRadius / (geometricAltitude + self.earthRadius)
        return geopotentialAltitude
    
    def _Gravity(self, geometricAltitude): #meters
        graviationalAcceleration = self.gravitationalAcceleration * (self.earthRadius / (self.earthRadius + geometricAltitude)) ** 2
        return graviationalAcceleration

    def _Temperature(self, index, geopotentialAltitude):
        scalar = self.lapseRates[index] * (geopotentialAltitude - self.segments[index]) / 1000
        return scalar
    
    def _baseTemperatures(self):
        baseTemperatures = [self.standardTemperature]
        for i in range(len(self.segments) - 1):
            baseTemperatures.append(baseTemperatures[i] + self._Temperature(i, self.segments[i + 1]))
        return baseTemperatures

    def Temperature(self, index, geometricAltitude):
        temperature = self._base_temps[index] + self._Temperature(index, self._Altitude(geometricAltitude))
        return temperature
    
    def _Pressure(self, index, geopotentialAltitude):
        if self.lapseRates[index] != 0:
            scalar = (self._base_temps[index] / ((self._base_temps[index] + self.lapseRates[index] * (geopotentialAltitude - self.segments[index]) / 1000))) ** (self.gravitationalAcceleration * self.mixedMolecularWeight / (self.gasConstant * self.lapseRates[index]))
        else:
            scalar = np.exp((-self.gravitationalAcceleration * self.mixedMolecularWeight * (geopotentialAltitude - self.segments[index]) / 1000) / (self.gasConstant * self._base_temps[index]))
        return scalar
    
    def _basePressures(self):
        basePressures = [self.standardPressure]
        for i in range(len(self.segments) - 1):
            basePressures.append(basePressures[i] * self._Pressure(i, self.segments[i + 1]))
        return basePressures

    def Pressure(self, index, geometricAltitude):
        pressure = self._base_pressures[index] * self._Pressure(index, self._Altitude(geometricAltitude))
        return pressure

    def Density(self, pressure, temperature):
        density = pressure * self.mixedMolecularWeight / (self.gasConstant * temperature)
        return density
    
    def _Qualities(self, geometricAltitude):
        i = bisect.bisect(self.segments, geometricAltitude) - 1
        pressure = self.Pressure(i, geometricAltitude)
        temperature = self.Temperature(i, geometricAltitude)
        density = self.Density(pressure, temperature)
        gravity = self._Gravity(geometricAltitude)
        return pressure, temperature, density, gravity