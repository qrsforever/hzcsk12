#!/bin/bash

firewall-cmd --add-port=8119/tcp --permanent
firewall-cmd --add-port=9119/tcp --permanent

# firewall-cmd --add-port=8139/tcp --permanent
# firewall-cmd --add-port=8179/tcp --permanent

firewall-cmd --reload
