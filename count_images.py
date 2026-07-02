import os

bus_folder="dataset/bus"
background_folder="dataset/background"

bus_count=len(os.listdir(bus_folder))
background_count=len(
    os.listdir(background_folder)
)

print(
    "Bus images:",
    bus_count
)

print(
    "Background images:",
    background_count
)

print(
    "Total:",
    bus_count+background_count
)