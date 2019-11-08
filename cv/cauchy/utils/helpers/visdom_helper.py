import numpy as np
import visdom
import socket
from psutil import process_iter
from signal import SIGTERM, SIGKILL


# create an plot window
def create_plot_window(vis, xlabel, ylabel, title):
    return vis.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(xlabel=xlabel, ylabel=ylabel, title=title),
    )


def create_text_window(vis, text):
    return vis.text(text)


def find_free_port(num_start, num_end):
    """find free port on localhost
  
  Returns:
    int: port number of free port
  """
    assert int(num_end) > int(num_start)
    for port in range(num_start, num_end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            res = sock.connect_ex(("0.0.0.0", port))
            if res != 0:
                return port


def free_port(port_number):
    for proc in process_iter():
        for conns in proc.connections(kind="inet"):
            if conns.laddr.port == port_number:
                proc.send_signal(SIGKILL)
                continue
