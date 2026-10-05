import mmap
import os
from multiprocessing.reduction import DupFd, ForkingPickler


def send(connection, value):
    with os.fdopen(os.memfd_create("tensorlane"), "w+b") as message:
        ForkingPickler(message).dump(value)
        message.flush()
        connection.send(DupFd(message.fileno()))


def recv(connection):
    with os.fdopen(connection.recv().detach(), "rb") as message:
        with mmap.mmap(message.fileno(), 0, access=mmap.ACCESS_READ) as data:
            return ForkingPickler.loads(data)
