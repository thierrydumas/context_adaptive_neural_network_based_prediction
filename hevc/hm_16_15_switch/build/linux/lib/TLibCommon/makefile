CONFIG = LIBRARY

SRC_DIR = ../../../../source/Lib/TLibCommon
INC_DIR = ../../../../source/Lib
LIB_DIR	= ../../../../lib
BIN_DIR	= ../../../../bin
SRC_DIR1 = ../../../../source/Lib/libmd5

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

USER_INC_DIRS = \
-I$(SRC_DIR) \
-I$(SRC_DIR2) \
-I$(DIRROOT) \
-I$(DIRGENPROTO) \
-I$(DIRPROTOBUFSRC) \
-I$(DIREIGEN) \
-I$(DIRNSYNC)

USER_LIB_DIRS = -L$(DIRLIBTF)

ifeq ($(HIGHBITDEPTH), 1)
HBD = HighBitDepth
else
HBD =
endif

OBJ_DIR	= ./objects$(HBD)
PRJ_NAME = TLibCommon$(HBD)

MAJOR_VER = 0
MINOR_VER = 1
VER	= $(MAJOR_VER).$(MINOR_VER)
DEFS = -DMSYS_LINUX

OBJS = \
$(OBJ_DIR)/ContextModel.o \
$(OBJ_DIR)/ContextModel3DBuffer.o \
$(OBJ_DIR)/Debug.o \
$(OBJ_DIR)/extraction_context.o \
$(OBJ_DIR)/integration_prediction_neural_network.o \
$(OBJ_DIR)/interface_c_python.o \
$(OBJ_DIR)/libmd5.o \
$(OBJ_DIR)/SEI.o \
$(OBJ_DIR)/TComCABACTables.o \
$(OBJ_DIR)/TComSampleAdaptiveOffset.o \
$(OBJ_DIR)/TComBitStream.o \
$(OBJ_DIR)/TComChromaFormat.o \
$(OBJ_DIR)/TComDataCU.o \
$(OBJ_DIR)/TComLoopFilter.o \
$(OBJ_DIR)/TComMotionInfo.o \
$(OBJ_DIR)/TComPattern.o \
$(OBJ_DIR)/TComPic.o \
$(OBJ_DIR)/TComPicSym.o \
$(OBJ_DIR)/TComPicYuvMD5.o \
$(OBJ_DIR)/TComPrediction.o \
$(OBJ_DIR)/TComRdCost.o \
$(OBJ_DIR)/TComRom.o \
$(OBJ_DIR)/TComSlice.o \
$(OBJ_DIR)/TComTrQuant.o \
$(OBJ_DIR)/TComTU.o \
$(OBJ_DIR)/TComInterpolationFilter.o \
$(OBJ_DIR)/TComWeightPrediction.o \
$(OBJ_DIR)/TComRdCostWeightPrediction.o \
$(OBJ_DIR)/TComPicYuv.o \
$(OBJ_DIR)/TComYuv.o \
$(OBJ_DIR)/tools.o \
$(OBJ_DIR)/visualization_debugging.o

LIBS = \
-lpthread \
-ltensorflow-core

DEBUG_LIBS =
RELEASE_LIBS =
STAT_LIBS = 
DYN_LIBS = -ldl

STAT_DEBUG_LIBS =
STAT_RELEASE_LIBS =
DYN_DEBUG_LIBS =
DYN_RELEASE_LIBS =

MAKE_FILE_NAME = ../../common/makefile.base
include $(MAKE_FILE_NAME)
