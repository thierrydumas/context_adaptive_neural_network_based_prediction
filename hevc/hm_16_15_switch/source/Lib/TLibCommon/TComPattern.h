/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2017, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __TCOMPATTERN__
#define __TCOMPATTERN__

#include "CommonDef.h"

#include "extraction_context.h"
#include "visualization_debugging.h"

class TComDataCU;
class TComTU;

class TComPatternParam
{
private:
    Pel* m_piROIOrigin;

public:
    Int m_iROIWidth;
    Int m_iROIHeight;
    Int m_iPatternStride;
    Int m_bitDepth;
  
    __inline Pel* getROIOrigin()
    {
        return m_piROIOrigin;
    }
    
    __inline const Pel* getROIOrigin() const
    {
        return m_piROIOrigin;
    }
    
    Void setPatternParamPel(Pel* piTexture,
                            Int iRoiWidth,
                            Int iRoiHeight,
                            Int iStride,
                            Int bitDepth);
};

class TComPattern
{
private:
    TComPatternParam  m_cPatternY;

public:
    Pel* getROIY() {return m_cPatternY.getROIOrigin();}
    const Pel* getROIY() const {return m_cPatternY.getROIOrigin();}
    Int getROIYWidth() const {return m_cPatternY.m_iROIWidth;}
    Int getROIYHeight() const {return m_cPatternY.m_iROIHeight;}
    Int getPatternLStride() const {return m_cPatternY.m_iPatternStride;}
    Int getBitDepthY() const {return m_cPatternY.m_bitDepth;}
    
    /** @brief Returns true if the pixel at the top-left corner of the masked context
      *        portion located above the current TU is inside the decoded channel.
      *
      * @param pcCU Current CU structure.
      * @param nbUnitsVerticalBetweenAnchorCurrentTU Number of 4x4 units (vertically) between
      *                                              the pixel at the top-left corner of the masked
      *                                              context portion located above the current TU
      *                                              and the pixel at the top-left of the current TU.
      * @param nbUnitsHorizontalBetweenAnchorCurrentTU Number of 4x4 units (horizontally) between
      *                                                the pixel at the top-left corner of the masked
      *                                                context portion located above the current TU
      *                                                and the pixel at the top-left of the current TU.
      * @param uiPartIdxLT Z-order index in the current CTU of the 4x4 unit located at
      *                    the top-left corner of the current TU.
      * @return Is the pixel at the top-left corner of the masked context portion located above the
      *         current TU inside the decoded channel?
      *
      */
    bool is_context_available(TComDataCU* pcCU,
                              const int& nbUnitsVerticalBetweenAnchorCurrentTU,
                              const int& nbUnitsHorizontalBetweenAnchorCurrentTU,
                              const int& uiPartIdxLT);
    
    Void initPattern(Pel* piY,
                     Int iRoiWidth,
                     Int iRoiHeight,
                     Int iStride,
                     Int bitDepthLuma);
};

#endif


