import subprocess
import sys
import os

def run_dash_app(script_name, port, base_path):
    env = os.environ.copy()
    env["DASH_PORT"] = str(port)
    env["DASH_BASE_PATH"] = base_path
    return subprocess.Popen([sys.executable, script_name], env=env)

if __name__ == "__main__":
    dashboards = [
        ("unbiasedness/unbiasedness_simulation.py", 8001, "/unbiasedness"),
        ("multicollinearity/multicollinearity.py", 8002, "/multicollinearity"),
        ("obv/obv.py", 8003, "/obv"),
        # ("hypothesis_testing/hypothesis_testing.py", 8004, "/hypothesis_testing"),
    ]

    processes = []
    for script_name, port, base_path in dashboards:
        process = run_dash_app(script_name, port, base_path)
        processes.append(process)

    input("Press Enter to terminate all dashboards...")

    for process in processes:
        process.terminate()

    print("All dashboards have been terminated.")
