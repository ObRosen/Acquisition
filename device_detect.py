import subprocess
import sys

import torch


def queryFreeGPU() -> int:
    try:
        deviceIndex = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs",
            shell=True
        ).decode()
        return int(deviceIndex)
    except Exception:
        print("failed to query free gpu", file=sys.stderr)
        return 0


def detectDevice() -> torch.device:
    if not torch.cuda.is_available():
        print("cuda device not available, use cpu")
        return torch.device("cpu")

    print("searching for free gpu...")

    # set cuda device
    if sys.platform == "win32":
        print("windows detected, use cuda:0")
        deviceIndex = 0  # not supporting detect free gpu on windows
    else:
        deviceIndex = queryFreeGPU()
        deviceIndex = int(deviceIndex)
        print("use device: cuda:", deviceIndex, "(GPU)")

    device = torch.device("cuda", deviceIndex)
    return device
