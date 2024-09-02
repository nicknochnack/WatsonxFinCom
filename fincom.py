import subprocess
from colorama import Fore, Back, init

init()

if __name__ == "__main__":
    print(Back.YELLOW + "Kicking off lineitem commentary" + Back.RESET)
    subprocess.run(["python", "pass1_flow.py"])
    print(Back.YELLOW + "Kicking off whole of business commentary" + Back.RESET)
    subprocess.run(["python", "pass2_flow.py"])
    print(Back.YELLOW + "Kicking off narrative commentary" + Back.RESET)
    subprocess.run(["python", "pass3_flow.py"])
