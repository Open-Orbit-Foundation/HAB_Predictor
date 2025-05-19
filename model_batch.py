from model import LaunchSite, Balloon, Payload, MissionProfile
from atmosphere import standardAtmosphere

launch_sites = [
    LaunchSite(1400)
]

balloons = [
    Balloon(0.35, 19.8 * 0.3048, 0.55, "Helium", 0.9),
    Balloon(0.35, 19.8 * 0.3048, 0.55, "Helium", 1.0),
    Balloon(0.35, 19.8 * 0.3048, 0.55, "Helium", 1.1),
    Balloon(0.35, 19.8 * 0.3048, 0.55, "Helium", 1.2),
    Balloon(0.35, 19.8 * 0.3048, 0.55, "Helium", 1.3),
    Balloon(0.35, 19.8 * 0.3048, 0.55, "Helium", 1.4),
    Balloon(0.35, 19.8 * 0.3048, 0.55, "Helium", 1.5),
    Balloon(0.35, 19.8 * 0.3048, 0.55, "Helium", 1.6),
    Balloon(0.35, 19.8 * 0.3048, 0.55, "Helium", 1.7),
    Balloon(0.35, 19.8 * 0.3048, 0.55, "Helium", 1.8),
    Balloon(0.35, 19.8 * 0.3048, 0.55, "Helium", 1.9),
    Balloon(0.35, 19.8 * 0.3048, 0.55, "Helium", 2.0),
    Balloon(0.6, 19.8 * 0.3048, 0.55, "Helium", 1.1),
    Balloon(0.6, 19.8 * 0.3048, 0.55, "Helium", 1.2),
    Balloon(0.6, 19.8 * 0.3048, 0.55, "Helium", 1.3),
    Balloon(0.6, 19.8 * 0.3048, 0.55, "Helium", 1.4),
    Balloon(0.6, 19.8 * 0.3048, 0.55, "Helium", 1.5),
    Balloon(0.6, 19.8 * 0.3048, 0.55, "Helium", 1.6),
    Balloon(0.6, 19.8 * 0.3048, 0.55, "Helium", 1.7),
    Balloon(0.6, 19.8 * 0.3048, 0.55, "Helium", 1.8),
    Balloon(0.6, 19.8 * 0.3048, 0.55, "Helium", 1.9),
    Balloon(0.6, 19.8 * 0.3048, 0.55, "Helium", 2.0)
]

payloads = [
    Payload(0.6, 2 * 0.3048, 1.2),
    Payload(0.3, 2 * 0.3048, 1.2)
]

mission_profiles = [
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[0], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[1], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[2], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[3], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[4], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[5], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[6], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[7], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[8], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[9], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[10], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[11], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[12], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[13], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[14], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[15], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[16], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[17], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[18], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[19], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[20], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[21], payloads[0])
]

flight_profiles = []