# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Top-level control of the building/installation/cleaning of various targets
#
# set empty to prevent any implicit rules from firing.
.SUFFIXES:

# where our installed python packages will live
VIRTUALENV_DIR := .venv
VIRTUALENV_EXE := virtualenv -p python3  # use pyvenv for python3 install
ACTIVATE := $(VIRTUALENV_DIR)/bin/activate

# get release version info
RELEASE := $(strip $(shell grep '^VERSION *=' setup.py | cut -f 2 -d '=' \
	                         | tr -d "\'"))

# basic check to see if any CUDA compatible GPU is installed
# set this to false to turn off GPU related functionality
HAS_GPU := $(shell nvcc --version > /dev/null 2>&1 && echo true)

ifdef HAS_GPU
# Get CUDA_ROOT for LD_RUN_PATH
export CUDA_ROOT:=$(patsubst %/bin/nvcc,%, $(realpath $(shell which nvcc)))
else
# Try to find CUDA.  Kernels will still need nvcc in path
export CUDA_ROOT:=$(firstword $(wildcard $(addprefix /usr/local/, cuda-7.5 cuda-7.0 cuda)))

ifdef CUDA_ROOT
export PATH:=$(CUDA_ROOT)/bin:$(PATH)
HAS_GPU := $(shell $(CUDA_ROOT)/bin/nvcc --version > /dev/null 2>&1 && echo true)
endif
endif
ifdef CUDA_ROOT
# Compiling with LD_RUN_PATH eliminates the need for LD_LIBRARY_PATH
# when running
export LD_RUN_PATH:=$(CUDA_ROOT)/lib64
endif

default: env

env: $(ACTIVATE)

$(ACTIVATE): requirements.txt gpu_requirements.txt vis_requirements.txt
	@echo "Updating virtualenv dependencies in: $(VIRTUALENV_DIR)..."
	@test -d $(VIRTUALENV_DIR) || $(VIRTUALENV_EXE) $(VIRTUALENV_DIR)
	@. $(ACTIVATE); pip install -U pip
	@# cython added separately due to h5py dependency ordering bug.  See:
	@# https://github.com/h5py/h5py/issues/535
	@. $(ACTIVATE); pip install cython==0.23.1
	@. $(ACTIVATE); pip install -r requirements.txt
ifeq ($(VIS), true)
	@echo "Updating visualization related dependecies in $(VIRTUALENV_DIR)..."
	@. $(ACTIVATE); pip install -r vis_requirements.txt
endif
	@echo
ifeq ($(HAS_GPU), true)
	@echo "Updating GPU dependencies in $(VIRTUALENV_DIR)..."
	@. $(ACTIVATE); pip install -r gpu_requirements.txt
	@echo
endif
	@echo "Installing neon in development mode..."
	@. $(ACTIVATE); python setup.py develop
	@echo "######################"
	@echo "Setup complete.  Type:"
	@echo "    . '$(ACTIVATE)'"
	@echo "to work interactively"
	@echo "######################"
	@touch $(ACTIVATE)
	@echo

