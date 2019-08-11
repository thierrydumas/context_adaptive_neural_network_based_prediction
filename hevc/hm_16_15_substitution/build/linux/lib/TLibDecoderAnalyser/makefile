CONFIG = LIBRARY

SRC_DIR	= ../../../../source/Lib/TLibDecoder
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

USER_INC_DIRS = \
-I$(SRC_DIR) \
-I$(SRC_DIR2) \
-I$(DIRROOT) \
-I$(DIRGENPROTO) \
-I$(DIRPROTOBUFSRC) \
-I$(DIREIGEN) \
-I$(DIRNSYNC)

USER_LIB_DIRS =

ifeq ($(HIGHBITDEPTH), 1)
HBD = HighBitDepth
else
HBD =
endif

OBJ_DIR	= ./objects$(HBD)
PRJ_NAME = TLibDecoderAnalyser$(HBD)
MAJOR_VER = 0
MINOR_VER = 1
VER = $(MAJOR_VER).$(MINOR_VER)
DEFS = -DMSYS_LINUX -DRExt__DECODER_DEBUG_BIT_STATISTICS=1

OBJS = \
$(OBJ_DIR)/AnnexBread.o \
$(OBJ_DIR)/NALread.o \
$(OBJ_DIR)/SEIread.o \
$(OBJ_DIR)/SyntaxElementParser.o \
$(OBJ_DIR)/TDecBinCoderCABAC.o \
$(OBJ_DIR)/TDecCAVLC.o \
$(OBJ_DIR)/TDecCu.o \
$(OBJ_DIR)/TDecEntropy.o \
$(OBJ_DIR)/TDecGop.o \
$(OBJ_DIR)/TDecSbac.o \
$(OBJ_DIR)/TDecSlice.o \
$(OBJ_DIR)/TDecTop.o \
$(OBJ_DIR)/TDecConformance.o

LIBS = -lpthread
DEBUG_LIBS =
RELEASE_LIBS =
STAT_LIBS =
DYN_LIBS = -ldl
STAT_DEBUG_LIBS	=
STAT_RELEASE_LIBS =
DYN_DEBUG_LIBS =
DYN_RELEASE_LIBS =

MAKE_FILE_NAME = ../../common/makefile.base
include $(MAKE_FILE_NAME)
