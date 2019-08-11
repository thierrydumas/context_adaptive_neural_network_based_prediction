# `DIRROOT` is the absolute path to the directory containing the Tensorflow repository.
# In `$(MAKEFILE_LIST)`, the last item is the last included Makefile.
DIRROOT := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))/../../../tensorflow-1.9.0


