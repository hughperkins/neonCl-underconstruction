#!/bin/bash

SYSTEM=$(uname -s)

if [[ ${SYSTEM} == Linux ]]; then {
   # checks of /etc/os-release etc from https://github.com/torch/distro
    if [[ -r /etc/os-release ]]; then
        # this will get the required information without dirtying any env state
        DIST_VERS="$( ( . /etc/os-release &>/dev/null
                        echo "$ID $VERSION_ID") )"
        DISTRO="${DIST_VERS%% *}" # get our distro name
        VERSION="${DIST_VERS##* }" # get our version number
    elif [[ -r /etc/redhat-release ]]; then
        DIST_VERS=( $( cat /etc/redhat-release ) ) # make the file an array
        DISTRO="${DIST_VERS[0],,}" # get the first element and get lcase
        VERSION="${DIST_VERS[2]}" # get the third element (version)
    elif [[ -r /etc/lsb-release ]]; then
        DIST_VERS="$( ( . /etc/lsb-release &>/dev/null
                        echo "${DISTRIB_ID,,} $DISTRIB_RELEASE") )"
        DISTRO="${DIST_VERS%% *}" # get our distro name
        VERSION="${DIST_VERS##* }" # get our version number
    else # well, I'm out of ideas for now
        echo '==> Failed to determine distro and version.'
        exit 1
    fi

  if [[ ${DISTRO} == ubuntu ]]; then {
    sudo apt-get install -y libhdf5-dev libyaml-dev pkg-config python-virtualenv libpython2.7-dev
  } else {
    echo distro ${DISTRO} not yet implemented in install-deps, please install manually using http://neon.nervanasys.com/docs/latest/installation.html
  } fi;
} else {
  echo os ${SYSTEM} not yet implemented in install-deps
  echo Please install manually using http://neon.nervanasys.com/docs/latest/installation.html
} fi

