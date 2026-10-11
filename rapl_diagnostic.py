import subprocess
import time


ENERGY_COUNTER = r"\Energy Meter(RAPL_Package0_PKG)\Energy"
POWER_COUNTER = r"\Energy Meter(RAPL_Package0_PKG)\Power"
TIME_COUNTER = r"\Energy Meter(RAPL_Package0_PKG)\Time"


def get_counter(counter_path):
    """
    Read one Windows performance counter using PowerShell.
    """

    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        f"(Get-Counter '{counter_path}').CounterSamples[0].CookedValue"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    return float(result.stdout.strip())


print("=" * 55)
print(" RAPL ENERGY COUNTER DIAGNOSTIC")
print("=" * 55)

print("\nReading initial values...\n")

energy_1 = get_counter(ENERGY_COUNTER)
power_1 = get_counter(POWER_COUNTER)
time_1 = get_counter(TIME_COUNTER)

print(f"Energy  : {energy_1}")
print(f"Power   : {power_1}")
print(f"Time    : {time_1}")

print("\nWaiting 5 seconds...")

time.sleep(5)

print("\nReading final values...\n")

energy_2 = get_counter(ENERGY_COUNTER)
power_2 = get_counter(POWER_COUNTER)
time_2 = get_counter(TIME_COUNTER)

print(f"Energy  : {energy_2}")
print(f"Power   : {power_2}")
print(f"Time    : {time_2}")

print("\n" + "=" * 55)
print(" CHANGES")
print("=" * 55)

energy_change = energy_2 - energy_1
time_change = time_2 - time_1

print(f"\nRaw Energy Change : {energy_change}")
print(f"Counter Time Change : {time_change}")

if time_change != 0:
    print(
        f"\nRaw Energy Change / Counter Time = "
        f"{energy_change / time_change}"
    )

print("\n" + "=" * 55)
print(" END")
print("=" * 55)