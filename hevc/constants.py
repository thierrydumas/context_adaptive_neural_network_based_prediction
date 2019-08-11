"""A library defining the paths to the different HEVC executables."""

import os
import sys

if sys.platform in ('win32', 'cygwin'):
    
    # The paths below are used when Visual Studio 2017 builds
    # the different HEVCs binaries.
    PATH_TO_EXE_ENCODER_REGULAR = 'hevc/hm_16_15_regular/bin/vc2015/x64/Release/TAppEncoder.exe'
    PATH_TO_EXE_DECODER_REGULAR = 'hevc/hm_16_15_regular/bin/vc2015/x64/Release/TAppDecoder.exe'
    PATH_TO_EXE_ENCODER_SUBSTITUTION = 'hevc/hm_16_15_substitution/bin/vc2015/x64/Release/TAppEncoder.exe'
    PATH_TO_EXE_DECODER_SUBSTITUTION = 'hevc/hm_16_15_substitution/bin/vc2015/x64/Release/TAppDecoder.exe'
    PATH_TO_EXE_ENCODER_SWITCH = 'hevc/hm_16_15_switch/bin/vc2015/x64/Release/TAppEncoder.exe'
    PATH_TO_EXE_DECODER_SWITCH = 'hevc/hm_16_15_switch/bin/vc2015/x64/Release/TAppDecoder.exe'
else:
    PATH_TO_EXE_ENCODER_REGULAR = 'hevc/hm_16_15_regular/bin/TAppEncoderStatic'
    PATH_TO_EXE_DECODER_REGULAR = 'hevc/hm_16_15_regular/bin/TAppDecoderStatic'
    PATH_TO_EXE_ENCODER_SUBSTITUTION = 'hevc/hm_16_15_substitution/bin/TAppEncoderStatic'
    PATH_TO_EXE_DECODER_SUBSTITUTION = 'hevc/hm_16_15_substitution/bin/TAppDecoderStatic'
    PATH_TO_EXE_ENCODER_SWITCH = 'hevc/hm_16_15_switch/bin/TAppEncoderStatic'
    PATH_TO_EXE_DECODER_SWITCH = 'hevc/hm_16_15_switch/bin/TAppDecoderStatic'


