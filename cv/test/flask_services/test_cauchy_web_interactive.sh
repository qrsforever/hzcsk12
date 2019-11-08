#!/bin/bash

# uninstall cauchy
pip uninstall -y cauchy
pip uninstall -y vulkan
easy_install -U  -i http://116.85.55.108:6500/simple/ vulkan


# update package
cd ../../
python setup.py install

cd test/flask_services
python cauchy_services.py --port 8139