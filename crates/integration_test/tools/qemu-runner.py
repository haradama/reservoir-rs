#!/usr/bin/env python3
import sys
import subprocess

if len(sys.argv) < 2:
    print("Usage: qemu-runner.py <qemu-command> [args...] <firmware-file>")
    sys.exit(1)

command = sys.argv[1:]

print(f"--- RUNNING COMMAND: {' '.join(command)} ---")

try:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break

        if line:
            print(line, end="")

            if "EMULATOR_EXIT" in line:
                process.terminate()
                print("\n--- DETECTED EXIT SIGNAL (SUCCESS) ---")
                sys.exit(0)

    return_code = process.poll()
    sys.exit(return_code)

except KeyboardInterrupt:
    process.terminate()
    sys.exit(130)
except FileNotFoundError:
    print(f"Error: Command not found: {command[0]}")
    sys.exit(1)
