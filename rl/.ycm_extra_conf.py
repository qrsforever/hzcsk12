import os.path as p

DIR_OF_THIS_PROJECT = p.abspath(p.join(p.dirname(__file__), '..'))


def PythonSysPath(**kwargs):
    sys_path = kwargs['sys_path']

    dependencies = [
        p.join(DIR_OF_THIS_PROJECT, 'rl', 'app'),
        p.join(DIR_OF_THIS_PROJECT, 'rl', 'rlpyt'),
    ]

    sys_path[0:0] = dependencies

    return sys_path
