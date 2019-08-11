CONFIG = LIBRARY

SRC_DIR = ../../../../source/Lib/TLibEncoder
INC_DIR	= ../../../../source/Lib
LIB_DIR	= ../../../../lib
BIN_DIR	= ../../../../bin
SRC_DIR1 =
SRC_DIR2 = ../../../../../hm_common/c++/source_common
SRC_DIR3 =
SRC_DIR4 =

USER_INC_DIRS = \
-I$(SRC_DIR) \
-I$(SRC_DIR2)

USER_LIB_DIRS =

ifeq ($(HIGHBITDEPTH), 1)
HBD = HighBitDepth
else
HBD =
endif

OBJ_DIR	= ./objects$(HBD)
PRJ_NAME = TLibEncoder$(HBD)

MAJOR_VER = 0
MINOR_VER = 1
VER	= $(MAJOR_VER).$(MINOR_VER)
DEFS = -DMSYS_LINUX

OBJS = \
$(OBJ_DIR)/NALwrite.o \
$(OBJ_DIR)/SEIEncoder.o \
$(OBJ_DIR)/SEIwrite.o \
$(OBJ_DIR)/SyntaxElementWriter.o \
$(OBJ_DIR)/TEncBinCoderCABAC.o \
$(OBJ_DIR)/TEncBinCoderCABACCounter.o \
$(OBJ_DIR)/TEncSampleAdaptiveOffset.o \
$(OBJ_DIR)/TEncCavlc.o \
$(OBJ_DIR)/TEncCu.o \
$(OBJ_DIR)/TEncEntropy.o \
$(OBJ_DIR)/TEncGOP.o \
$(OBJ_DIR)/TEncSbac.o \
$(OBJ_DIR)/TEncSearch.o \
$(OBJ_DIR)/TEncSlice.o \
$(OBJ_DIR)/TEncTop.o \
$(OBJ_DIR)/TEncPic.o \
$(OBJ_DIR)/TEncPreanalyzer.o \
$(OBJ_DIR)/TEncRateCtrl.o \
$(OBJ_DIR)/tools.o \
$(OBJ_DIR)/visualization_debugging.o \
$(OBJ_DIR)/WeightPredAnalysis.o

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
