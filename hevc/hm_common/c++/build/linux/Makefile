CC := gcc

# -g adds debugging information to the executable file.
# -Wall turns on several compiler warnings.
CCFLAGS := -g -Wall -std=c++11

# The included file "../../../common.mk" defines `DIRROOT`.
include ../../../common.mk

DIRMAKEFILE := $(DIRROOT)/tensorflow/contrib/makefile

# `DIRGENPROTO` is the path to the directory containing the file "tensorflow/core/framework/graph.pb.h".
DIRGENPROTO := $(DIRMAKEFILE)/gen/proto

# `DIRPROTOBUFSRC` is the path to the directory containing the file "google/protobuf/arena.h".
DIRPROTOBUFSRC := $(DIRMAKEFILE)/downloads/protobuf/src

# `DIREIGEN` is the path to the directory containing the file "usupported/Eigen/CXX11/Tensor".
DIREIGEN := $(DIRMAKEFILE)/downloads/eigen

# `DIRNSYNC` is the path to the directory containing the file "nsync_cv.h".
DIRNSYNC := $(DIRMAKEFILE)/downloads/nsync/public

# `DIRLIBTF` is the path to the directory containing "libtensorflow-core.a".
DIRLIBTF := $(DIRMAKEFILE)/gen/lib

# `DIRLIBPROTOBUF` is the path to the directory containing "libprotobuf.so".
DIRLIBPROTOBUF := $(DIRPROTOBUFSRC)/.libs

# `DIRLIBNSYNC` is the path to the directory containing "libnsync.a".
DIRLIBNSYNC := $(DIRMAKEFILE)/downloads/nsync/builds/default.linux.c++11

INCLUDES := \
-I$(DIRROOT) \
-I$(DIRGENPROTO) \
-I$(DIRPROTOBUFSRC) \
-I$(DIREIGEN) \
-I$(DIRNSYNC) \
-I../../source_common

LDFLAGS := \
-L$(DIRLIBTF) \
-L$(DIRLIBPROTOBUF) \
-L$(DIRLIBNSYNC)

# "libpython2.7.so" is necessary for the interface
# between C and Python.
LIBS := \
-Wl,--allow-multiple-definition \
-Wl,--whole-archive \
-ltensorflow-core \
-Wl,--no-whole-archive \
-lpython2.7 \
-lstdc++ \
-lprotobuf \
-lm \
-ldl \
-lpthread \
-lnsync

EXE := ../../bin/executable

# `DIROBJS` is the path to the directory containing all the
# compiled objects.
DIROBJS := ../../bin/objs

NAMES := \
extraction_context.o \
integration_prediction_neural_network.o \
interface_c_python.o \
main.o \
tests.o \
tools.o \
visualization_debugging.o

# `OBJS` contains the path to each compiled object.
OBJS := $(addprefix $(DIROBJS)/, $(NAMES))

# By default, Makefile targets are "file targets": they
# are used to build files from other files. ".PHONY" tells
# "make" that "clean" and "printvars" are not files.
.PHONY: clean printvars

# Typing "make" invokes the 1st argument entry. This
# target entry can have any name. "all" is a convention.
all: $(EXE)

$(EXE): $(OBJS)
	$(CC) $(CCFLAGS) $(INCLUDES) -o $(EXE) $(OBJS) $(LDFLAGS) $(LIBS)

# `$@` is the automatic variable that holds the name of the target.
$(DIROBJS)/%.o: ../../source_common/%.cpp
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

$(DIROBJS)/%.o: ../../source_test/%.cpp
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm ../../bin/executable --force
	rm ../../bin/objs/* --force

printvars:
	@echo $(DIRROOT)


