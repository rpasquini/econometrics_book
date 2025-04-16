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
        ("unbiasedness/unbiasedness_simulation.py", 8001),
        ("multicollinearity/multicollinearity.py", 8002),
        ("obv/obv.py", 8003),
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
