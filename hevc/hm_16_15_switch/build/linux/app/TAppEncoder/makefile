CONFIG = CONSOLE

SRC_DIR	= ../../../../source/App/TAppEncoder
INC_DIR	= ../../../../source/Lib
LIB_DIR	= ../../../../lib
BIN_DIR	= ../../../../bin
SRC_DIR1 =

# The code for integrating the prediction neural networks
# is in the directory at `SRC_DIR2`.
SRC_DIR2 = ../../../../../hm_common/c++/source_common
SRC_DIR3 =
SRC_DIR4 =

# The file "../../../../../hm_common/common.mk" defines `DIRROOT`.
include ../../../../../hm_common/common.mk

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

USER_INC_DIRS = \
-I$(SRC_DIR) \
-I$(SRC_DIR2) \
-I$(DIRROOT) \
-I$(DIRGENPROTO) \
-I$(DIRPROTOBUFSRC) \
-I$(DIREIGEN) \
-I$(DIRNSYNC)

USER_LIB_DIRS = \
-L$(DIRLIBTF) \
-L$(DIRLIBPROTOBUF) \
-L$(DIRLIBNSYNC)

ifeq ($(HIGHBITDEPTH), 1)
HBD = HighBitDepth
else
HBD =
endif

OBJ_DIR = ./objects$(HBD)
PRJ_NAME = TAppEncoder$(HBD)
DEFS = -DMSYS_LINUX -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 -DMSYS_UNIX_LARGEFILE

OBJS = \
$(OBJ_DIR)/encmain.o \
$(OBJ_DIR)/TAppEncCfg.o \
$(OBJ_DIR)/TAppEncTop.o

# Thanks to -Wl,--allow-multiple-definition -Wl,--whole-archive
# the global constructors that register factories and kernels
# are not stripped out.
LIBS = \
-lprotobuf \
-ldl \
-Wl,--allow-multiple-definition \
-Wl,--whole-archive \
-ltensorflow-core \
-Wl,--no-whole-archive \
-lpython2.7

# If `LINKASAN` is defined, ASAN is linked to the executable.
ifdef LINKASAN
LIBS += -lasan
endif

DEBUG_LIBS =
RELEASE_LIBS =
STAT_LIBS = -lpthread
DYN_LIBS =

DYN_DEBUG_LIBS = \
-lTLibEncoder$(HBD)d \
-lTLibCommon$(HBD)d \
-lTLibVideoIO$(HBD)d \
-lTAppCommon$(HBD)d

DYN_DEBUG_PREREQS = \
$(LIB_DIR)/libTLibEncoder$(HBD)d.a \
$(LIB_DIR)/libTLibCommon$(HBD)d.a \
$(LIB_DIR)/libTLibVideoIO$(HBD)d.a \
$(LIB_DIR)/libTAppCommon$(HBD)d.a

STAT_DEBUG_LIBS = \
-lTLibEncoder$(HBD)Staticd \
-lTLibCommon$(HBD)Staticd \
-lTLibVideoIO$(HBD)Staticd \
-lTAppCommon$(HBD)Staticd \
-lnsync

STAT_DEBUG_PREREQS = \
$(LIB_DIR)/libTLibEncoder$(HBD)Staticd.a \
$(LIB_DIR)/libTLibCommon$(HBD)Staticd.a \
$(LIB_DIR)/libTLibVideoIO$(HBD)Staticd.a \
$(LIB_DIR)/libTAppCommon$(HBD)Staticd.a

DYN_RELEASE_LIBS = \
-lTLibEncoder$(HBD) \
-lTLibCommon$(HBD) \
-lTLibVideoIO$(HBD) \
-lTAppCommon$(HBD)

DYN_RELEASE_PREREQS	= \
$(LIB_DIR)/libTLibEncoder$(HBD).a \
$(LIB_DIR)/libTLibCommon$(HBD).a \
$(LIB_DIR)/libTLibVideoIO$(HBD).a \
$(LIB_DIR)/libTAppCommon$(HBD).a

STAT_RELEASE_LIBS = \
-lTLibEncoder$(HBD)Static \
-lTLibCommon$(HBD)Static \
-lTLibVideoIO$(HBD)Static \
-lTAppCommon$(HBD)Static \
-lnsync

STAT_RELEASE_PREREQS = \
$(LIB_DIR)/libTLibEncoder$(HBD)Static.a \
$(LIB_DIR)/libTLibCommon$(HBD)Static.a \
$(LIB_DIR)/libTLibVideoIO$(HBD)Static.a \
$(LIB_DIR)/libTAppCommon$(HBD)Static.a

ifeq ($(EXTENSION_360_VIDEO), 1)
  DYN_DEBUG_LIBS += -lTAppEncHelper360$(HBD)d -lTLib360$(HBD)d
  DYN_DEBUG_PREREQS += $(LIB_DIR)/libTAppEncHelper360$(HBD)d.a $(LIB_DIR)/libTLib360$(HBD)d.a
  STAT_DEBUG_LIBS += -lTAppEncHelper360$(HBD)Staticd -lTLib360$(HBD)Staticd
  STAT_DEBUG_PREREQS += $(LIB_DIR)/libTAppEncHelper360$(HBD)Staticd.a $(LIB_DIR)/libTLib360$(HBD)Staticd.a
  DYN_RELEASE_LIBS += -lTAppEncHelper360$(HBD) -lTLib360$(HBD)
  DYN_RELEASE_PREREQS += $(LIB_DIR)/libTAppEncHelper360$(HBD).a $(LIB_DIR)/libTLib360$(HBD).a
  STAT_RELEASE_LIBS += -lTAppEncHelper360$(HBD)Static -lTLib360$(HBD)Static
  STAT_RELEASE_PREREQS+= $(LIB_DIR)/libTAppEncHelper360$(HBD)Static.a $(LIB_DIR)/libTLib360$(HBD)Static.a
endif

MAKE_FILE_NAME = ../../common/makefile.base
include $(MAKE_FILE_NAME)
