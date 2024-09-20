import subprocess
import sys
import os


def run_dash_app(script_name, port):
    env = os.environ.copy()
    env["DASH_PORT"] = str(port)
    return subprocess.Popen([sys.executable, script_name], env=env)


if __name__ == "__main__":
    # Define the dashboards and their ports
    dashboards = [
        ("unbiasedness_simulation.py", 8050),
        ("unbiasedness_simulation_copy.py", 8060),
    ]

    # Start processes for each dashboard
    processes = []
    for script_name, port in dashboards:
        process = run_dash_app(script_name, port)
        processes.append(process)

    # Wait for user input to terminate
    input("Press Enter to terminate all dashboards...")

    # Terminate all processes
    for process in processes:
        process.terminate()

    print("All dashboards have been terminated.")
