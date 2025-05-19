from utils import Geometry, Integrator, Utility
from atmosphere import standardAtmosphere
from dataclasses import dataclass
from functools import partial
import numpy as np
import math
import sys

@dataclass(frozen=False)
class LaunchSite:
    altitude: float
    #latitude: float
    #longitude: float

@dataclass(frozen=False)
class Balloon:
    mass: float
    burst_diameter: float
    drag_coefficient: float
    gas: str
    gas_volume: float
    gas_moles: float = 0.0

    def __post_init__(self):
        self.gas_moles = self.gas_volume * 1000 / ((1.380622 * 6.022169) * 273.15 / 101.325)

@dataclass(frozen=False)
class Payload:
    mass: float
    parachute_diameter: float
    parachute_drag_coefficient: float

@dataclass(frozen=False)
class MissionProfile:
    launch_site: LaunchSite
    balloon: Balloon
    payload: Payload

@dataclass(frozen=False)
class FlightProfile(MissionProfile):
    times: list[float]
    altitudes: list[float]
    velocities: list[float]
    accelerations: list[float]
    forces: list[float]
    pressures: list[float]
    temperatures: list[float]
    densities: list[float]
    gravities: list[float]
    burst_altitude: float
    max_altitude: float
    burst_time: float
    flight_time: float
    #latitudes: list[float]
    #longitudes: list[float]

class Model:
    helium_mm = 4.002602
    atmosphere = standardAtmosphere()

    def __init__(self, time_step, profiles, result):
        self.time_step = time_step
        self.profiles = profiles
        self.result = result

    def _ascent_acceleration(self, altitude, velocity, profile, mass):
        net_force = self._ascent_net_force(altitude, velocity, profile, mass)
        return net_force / mass
    
    def _ascent_net_force(self, altitude, velocity, profile, mass):
        #idx = np.searchsorted(self.atmosphere.altitudes_cache, altitude)
        #pressure, temperature, density, gravity = self.atmosphere.atmospheric_properties[idx]
        pressure, temperature, density, gravity = self.atmosphere._Qualities(altitude)
        volume = profile.balloon.gas_moles * (1.380622 * 6.022169) * temperature / pressure / 1000
        buoyant_force = density * gravity * volume
        weight_force = gravity * mass
        drag_force =  0.5 * density * velocity ** 2 * profile.balloon.drag_coefficient * Geometry.sphere_cross_section(volume) * np.sign(velocity)
        return buoyant_force - weight_force - drag_force
    
    def _descent_acceleration(self, altitude, velocity, profile, mass):
        net_force = self._descent_net_force(altitude, velocity, profile, mass)
        return net_force / mass
    
    def _descent_net_force(self, altitude, velocity, profile, mass):
        #idx = np.searchsorted(self.atmosphere.altitudes_cache, altitude)
        #pressure, temperature, density, gravity = self.atmosphere.atmospheric_properties[idx]
        _, _, density, gravity = self.atmosphere._Qualities(altitude)
        buoyant_force = 0
        weight_force = gravity * mass
        drag_force =  0.5 * density * velocity ** 2 * profile.payload.parachute_drag_coefficient * (profile.payload.parachute_diameter ** 2 * math.pi / 4) * np.sign(velocity)
        return buoyant_force - weight_force - drag_force

    def altitude_model(self, logging: bool = True, interval: int = 1):
        #start = time.perf_counter()
        if logging:
            padding = len(f" Modelling {len(self.profiles)} Profiles ...")
        for i, profile in enumerate(self.profiles):
            if logging:
                sys.stdout.write(f"""\033[?25l{Utility.progress_bar(i, len(self.profiles), prefix=f" Modelling Profile {i + 1} ...".rjust(padding), suffix="Complete", bar_length=100)}
                {Utility.progress_bar(0, 1, prefix=f"Ascent ...".rjust(padding), suffix="Complete", bar_length=100)}
                {Utility.progress_bar(0, 1, prefix=f"Descent ...".rjust(padding), suffix="Complete", bar_length=100)}\x1b[2F\n""")
                sys.stdout.flush()
            times = [0]
            logged_times = [0]
            altitude = profile.launch_site.altitude
            altitudes = [altitude]
            velocity = 0.01
            velocities = [velocity]
            accelerations = [0]
            forces = [0]
            #idx = np.searchsorted(self.atmosphere.altitudes_cache, altitudes[-1])
            #pressure, temperature, density, gravity = self.atmosphere.atmospheric_properties[idx]
            pressure, temperature, density, gravity = self.atmosphere._Qualities(altitudes[-1])
            pressures = [pressure]
            temperatures = [temperature]
            densities = [density]
            gravities = [gravity]
            burst_volume = (4 / 3) * math.pi * (profile.balloon.burst_diameter / 2) ** 3
            volume = profile.balloon.gas_moles * (1.380622 * 6.022169) * temperature / pressure / 1000
            ascent_mass = profile.payload.mass + profile.balloon.mass + (self.helium_mm * profile.balloon.gas_moles / 1000)
            descent_mass = profile.payload.mass
            while volume < burst_volume:
                altitude, velocity, acceleration = Integrator.rk2_second_order(altitude, velocity, partial(self._ascent_acceleration, profile = profile, mass = ascent_mass), self.time_step)
                if altitude <= altitudes[0]:
                     break
                #idx = np.searchsorted(self.atmosphere.altitudes_cache, altitude)
                #pressure, temperature, density, gravity = self.atmosphere.atmospheric_properties[idx]
                pressure, temperature, density, gravity = self.atmosphere._Qualities(altitude)
                volume = profile.balloon.gas_moles * (1.380622 * 6.022169) * temperature / pressure / 1000
                times.append(times[-1] + self.time_step)
                if len(times) % interval == 0:
                    logged_times.append(logged_times[-1] + interval * self.time_step)
                    altitudes.append(altitude)
                    velocities.append(velocity)
                    accelerations.append(acceleration)
                    forces.append(self._ascent_net_force(altitude, velocity, profile, ascent_mass))
                    times.append(times[-1] + self.time_step)
                    pressures.append(pressure)
                    temperatures.append(temperature)
                    densities.append(density)
                    gravities.append(gravity)
                    if logging:
                        sys.stdout.write(f"\x1b[K{Utility.progress_bar(volume, burst_volume, prefix=f"Ascent ...".rjust(padding), suffix="Complete", bar_length=100)}")
                        sys.stdout.flush()
            if logging:
                sys.stdout.write(f"{Utility.progress_bar(1, 1, prefix=f"Ascent ...".rjust(padding), suffix="Complete", bar_length=100)}\x1b[E")
                sys.stdout.flush()
            if altitude <= altitudes[0]:
                burst_altitude = float('nan')
                burst_time = float('nan')
            else:
                burst_altitude = altitude
                burst_time = times[-1]
            while altitude > altitudes[0]:
                altitude, velocity, acceleration = Integrator.rk2_second_order(altitude, velocity, partial(self._descent_acceleration, profile = profile, mass = descent_mass), self.time_step / 2)
                #idx = np.searchsorted(self.atmosphere.altitudes_cache, altitude)
                #pressure, temperature, density, gravity = self.atmosphere.atmospheric_properties[idx]
                pressure, temperature, density, gravity = self.atmosphere._Qualities(altitude)
                times.append(times[-1] + self.time_step / 2)
                if len(times) % interval == 0:
                    logged_times.append(logged_times[-1] + interval * self.time_step / 2)
                    accelerations.append(acceleration)
                    velocities.append(velocity)
                    altitudes.append(altitude)
                    forces.append(self._descent_net_force(altitude, velocity, profile, descent_mass))
                    pressures.append(pressure)
                    temperatures.append(temperature)
                    densities.append(density)
                    gravities.append(gravity)
                    if logging:
                        sys.stdout.write(f"\x1b[K{Utility.progress_bar(altitudes[0], altitude, prefix=f"Descent ...".rjust(padding), suffix="Complete", bar_length=100)}")
                        sys.stdout.flush()
            if logging:
                sys.stdout.write("\x1b[2F\x1b[J\033[?25h")
                sys.stdout.flush()
            '''if len(times) % 10 != 0:
                    logged_times.append(logged_times[-1] + (len(times) % 10) * self.time_step)
                    accelerations.append(acceleration)
                    velocities.append(velocity)
                    altitudes.append(altitude)
                    forces.append(self._net_force(altitude, velocity, profile, False))
                    pressures.append(pressure)
                    temperatures.append(temperature)
                    densities.append(density)
                    gravities.append(gravity)'''
            flight_time = times[-1]
            self.result.append(FlightProfile(profile.launch_site, profile.balloon, profile.payload, 
                                             logged_times, altitudes, velocities, accelerations, forces, 
                                             pressures, temperatures, densities, gravities,
                                             burst_altitude, np.max(altitudes), burst_time, flight_time))
        #end = time.perf_counter()
        #sys.stdout.write(f"{Utility.progress_bar(1, 1, prefix=f" Modelling {len(self.profiles)} Profiles ...".ljust(padding), suffix=f"Complete in {end - start:.2f} seconds\n", bar_length=100) if logging else print(f" Modelling {len(self.profiles)} Profiles in {end - start:.2f} seconds")}")
        #sys.stdout.flush()