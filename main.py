from model import LaunchSite, Balloon, Payload, MissionProfile, Model
import run
import numpy as np
import pandas as pd
import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.ticker as tic

def extract_attributes(obj, prefix=""):
    attributes = {}
    for attr in dir(obj):
        if not callable(getattr(obj, attr)) and not attr.startswith("__"):
            value = getattr(obj, attr)
            if hasattr(value, "__dict__"):
                nested_attrs = extract_attributes(value, prefix=f"{prefix}{attr}.")
                attributes.update(nested_attrs)
            else:
                attributes[f"{prefix}{attr}"] = value
    return attributes

def profiles_to_dataframe(profiles):
    profile_data = []
    for i, profile in enumerate(profiles):
        profile_dict = {"Profile ID": i}
        attributes = extract_attributes(profile)
        profile_dict.update(attributes)
        profile_data.append(profile_dict)
    profile_df = pd.DataFrame(profile_data)
    profile_df.set_index("Profile ID", inplace=True)
    return profile_df

def calculate_avg_velocities(profile):
    velocities = profile["velocities"]
    altitudes = profile["altitudes"]
    ascent_velocities = [v for v, _ in zip(velocities, altitudes) if v > 0]
    descent_velocities = [v for v, _ in zip(velocities, altitudes) if v < 0]
    avg_ascent = sum(ascent_velocities) / len(ascent_velocities) if ascent_velocities else float('nan')
    avg_descent = sum(descent_velocities) / len(descent_velocities) if descent_velocities else float('nan')
    return pd.Series([avg_ascent, avg_descent], index=["avg_ascent", "avg_descent"])

if __name__ == "__main__":
    launch_sites = [
        LaunchSite(1400)
    ]

    balloons = [
        Balloon(0.60, 6.03504, 0.55, "Helium", 2.12376), #75 cuft
        Balloon(0.15, 2.52, 0.55, "Helium", 0.9),
        Balloon(0.20, 3.00, 0.55, "Helium", 1.0),
        Balloon(0.30, 3.78, 0.55, "Helium", 1.2),
        Balloon(0.35, 4.12, 0.55, "Helium", 1.3),
        Balloon(0.60, 6.02, 0.55, "Helium", 1.5),
        Balloon(0.80, 7.00, 0.55, "Helium", 1.7),
        Balloon(1.00, 7.86, 0.55, "Helium", 1.9),
        Balloon(1.20, 8.63, 0.55, "Helium", 2.1),
        Balloon(1.50, 9.44, 0.55, "Helium", 2.3),
        Balloon(1.60, 9.71, 0.55, "Helium", 2.5),
        Balloon(1.80, 9.98, 0.55, "Helium", 2.7),
        Balloon(2.00, 10.54, 0.55, "Helium", 2.9),
        Balloon(3.00, 13.00, 0.55, "Helium", 4.0),
        Balloon(4.00, 12.06, 0.55, "Helium", 5.0)
    ]

    payloads = [
        Payload(0.952544, 3 * 0.3048, 0.97) #2.1 lb Rocketman profile
    ]

    mission_profiles = [
        MissionProfile(launch_sites[0], balloons[0], payloads[0]),
        MissionProfile(launch_sites[0], balloons[1], payloads[0]),
        MissionProfile(launch_sites[0], balloons[2], payloads[0]),
        MissionProfile(launch_sites[0], balloons[3], payloads[0]),
        MissionProfile(launch_sites[0], balloons[4], payloads[0]),
        MissionProfile(launch_sites[0], balloons[5], payloads[0]),
        MissionProfile(launch_sites[0], balloons[6], payloads[0]),
        MissionProfile(launch_sites[0], balloons[7], payloads[0]),
        MissionProfile(launch_sites[0], balloons[8], payloads[0]),
        MissionProfile(launch_sites[0], balloons[9], payloads[0]),
        MissionProfile(launch_sites[0], balloons[10], payloads[0]),
        MissionProfile(launch_sites[0], balloons[11], payloads[0]),
        MissionProfile(launch_sites[0], balloons[12], payloads[0]),
        MissionProfile(launch_sites[0], balloons[13], payloads[0])
    ]

    flight_profiles = []

    start = time.perf_counter()
    Model(1.1, mission_profiles, flight_profiles).altitude_model(True, 10) #RK4 < 1.5 (5.48s), RK2 < 1.1 (5.16s), RK1 < 1.0 (4.49s)
    #After Updates on 5/19: RK4 < 1.5 (4.18s), RK2 < 1.1 (3.59s), RK1 < 1.0 (2.90s)
    end= time.perf_counter()
    print(f"Singleprocessing {int(len(mission_profiles))} Profiles in {end - start:.2f} seconds.")

    '''start = time.perf_counter()
    with ProcessPoolExecutor(max_workers = 8) as executor:
        flight_profiles = list(executor.map(partial(run.predictor, dt = 1.1, logging = False, interval = 10), range(len(mission_profiles)), mission_profiles))
    end= time.perf_counter()
    print(f"Multiprocessing {int(len(mission_profiles))} Profiles in {end - start:.2f} seconds.")'''

    df = profiles_to_dataframe(flight_profiles)
    '''Index(['accelerations', 'altitudes', 'balloon.burst_diameter',
        'balloon.drag_coefficient', 'balloon.gas', 'balloon.gas_moles',
        'balloon.gas_volume', 'balloon.mass', 'burst_altitude', 'burst_time',
        'densities', 'flight_time', 'forces', 'gravities',
        'launch_site.altitude', 'max_altitude', 'payload.mass',
        'payload.parachute_diameter', 'payload.parachute_drag_coefficient',
        'pressures', 'temperatures', 'times',
        'velocities'],
        dtype='object')'''

    high_altitudes = df[["burst_altitude"]]
    print(high_altitudes)

    #max_altitudes = df[["max_altitude"]]
    #print(max_altitudes)

    final_velocities = df["velocities"].apply(lambda v: v[-1])
    print(final_velocities)

    # Apply the function to each row in the DataFrame
    average_velocities = df.apply(calculate_avg_velocities, axis=1)

    # Add the result to the summary DataFrame
    df[["avg_ascent", "avg_descent"]] = average_velocities

    # Display the updated DataFrame
    print(df[["avg_ascent", "avg_descent"]])

    fig, ax = plt.subplots(figsize = (12, 9))
    for i, profile in enumerate(flight_profiles):
        ax.plot(np.array(profile.times) / 3600, np.array(profile.altitudes) / 1000, label = f"Profile {i + 1}")
        ax.set_xlabel("Time (hr)")
        ax.xaxis.set_major_locator(tic.MultipleLocator(1))
        ax.xaxis.set_minor_locator(tic.AutoMinorLocator(5))
        ax.set_ylabel("Altitude (km)")
        ax.yaxis.set_major_locator(tic.MultipleLocator(5))
        ax.yaxis.set_minor_locator(tic.AutoMinorLocator(5))
        '''burst_index = profile.altitudes.index(max(profile.altitudes))
        burst_time = profile.times[burst_index]
        burst_altitude = profile.altitudes[burst_index]
        ax.plot(burst_time / 3600, burst_altitude / 1000, 'x', color='black')
        ax.annotate(
            f"{burst_altitude / 1000:.1f} km\n{burst_time / 3600:.1f} hr",
            xy = (burst_time / 3600, burst_altitude / 1000),
            xytext = (-20, 5),
            textcoords = 'offset points',
            ha = 'center',
            fontsize = 8,
            bbox = dict(facecolor = 'white', alpha = 0.7, edgecolor = 'none', boxstyle = 'round, pad = 0.1')
        )'''
    plt.title("RK4 Interial Model Altitude Profile(s)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.show()

    fig, ax = plt.subplots()
    for i, profile in enumerate(flight_profiles):
        ax.plot(np.array(profile.times) / 3600, np.array(profile.pressures) * 10, label = f"Profile {i + 1}")
        ax.set_xlabel("Time (hr)")
        ax.xaxis.set_major_locator(tic.MultipleLocator(1))
        ax.xaxis.set_minor_locator(tic.AutoMinorLocator(5))
        ax.set_ylabel("Pressure (hPa)")
        ax.set_yscale('log')
    plt.title("RK4 Interial Model Pressure Profile(s)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.show()