"""Owned Linux service exits when its controller dies, including SIGKILL of the parent."""
import ctypes
import os
import signal
import sys


def install_parent_guard(owner_pid):
    if owner_pid is None:return
    if not sys.platform.startswith("linux"):raise RuntimeError("Owned-service parent guard requires Linux")
    libc=ctypes.CDLL(None,use_errno=True)
    libc.prctl.argtypes=[ctypes.c_int]+[ctypes.c_ulong]*4
    libc.prctl.restype=ctypes.c_int
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError(ctypes.get_errno(),"PR_SET_PDEATHSIG failed")
    if os.getppid()!=owner_pid:raise SystemExit("Service controller already exited")
