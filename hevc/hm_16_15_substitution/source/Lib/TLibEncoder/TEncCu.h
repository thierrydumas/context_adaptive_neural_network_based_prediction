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

#ifndef __TENCCU__
#define __TENCCU__

#include "TLibCommon/CommonDef.h"
#include "TLibCommon/TComYuv.h"
#include "TLibCommon/TComPrediction.h"
#include "TLibCommon/TComTrQuant.h"
#include "TLibCommon/TComBitCounter.h"
#include "TLibCommon/TComDataCU.h"
#include "TEncEntropy.h"
#include "TEncSearch.h"
#include "TEncRateCtrl.h"

#include "visualization_debugging.h"

class TEncTop;
class TEncSbac;
class TEncCavlc;
class TEncSlice;

class TEncCu
{
private:

  TComDataCU** m_ppcBestCU;
  TComDataCU** m_ppcTempCU;
  UChar m_uhTotalDepth;
  TComYuv** m_ppcPredYuvBest;
  TComYuv** m_ppcResiYuvBest;
  TComYuv** m_ppcRecoYuvBest;
  TComYuv** m_ppcPredYuvTemp;
  TComYuv** m_ppcResiYuvTemp;
  TComYuv** m_ppcRecoYuvTemp;
  TComYuv** m_ppcOrigYuv;
  Bool m_bEncodeDQP;
  Bool m_bFastDeltaQP;
  Bool m_stillToCodeChromaQpOffsetFlag;
  Int m_cuChromaQpOffsetIdxPlus1;
  Int m_lumaLevelToDeltaQPLUT[LUMA_LEVEL_TO_DQP_LUT_MAXSIZE];
  Int m_lumaQPOffset;
  TEncSlice* m_pcSliceEncoder;
  TEncCfg* m_pcEncCfg;
  TEncSearch* m_pcPredSearch;
  TComTrQuant* m_pcTrQuant;
  TComRdCost* m_pcRdCost;
  TEncEntropy* m_pcEntropyCoder;
  TEncBinCABAC* m_pcBinCABAC;
  TEncSbac*** m_pppcRDSbacCoder;
  TEncSbac* m_pcRDGoOnSbacCoder;
  TEncRateCtrl* m_pcRateCtrl;

public:
  
  /** @brief Returns a pointer to an array of pointers to arrays
   *         for storing the number of times each mode is found in
   *         the fast list, for each width of target patch.
   *
   *  @return Pointer to the array of pointers to arrays.
   */
  UInt** getNbFastSelections() const;

  /** @brief Returns a pointer to an array of pointers to arrays
   *         for storing the number of times each mode wins the
   *         fast selection, for each width of target patch.
   *
   *  @return Pointer to the array of pointers to arrays.
   */
  UInt** getNbFastWins() const;

  /** @brief Returns a pointer to an array of pointers to arrays
   *         for storing the number of times each mode wins the
   *         rate-distortion selection, for each width of target
   *         patch.
   *
   *  @return Pointer to the array of pointers to arrays.
   */
  UInt** getNbRDWins() const;

  /** @brief Returns a pointer to an array for storing the number
   *         of times the pipeline {fast selection, RD selection}
   *         is run for each width of target patch.
   *
   *  @return Pointer to the array.
   */
  UInt* getNbFastRDRuns() const;

  /** @brief Returns a pointer to an array for storing the cumulated
   *         number of modes in the fast list for each width of target
   *         patch.
   *
   *  @return Pointer to the array.
   */
  UInt* getCumNbModesForRD() const;
  
  Void init(TEncTop* pcEncTop);
  Void setSliceEncoder(TEncSlice* pSliceEncoder) {m_pcSliceEncoder = pSliceEncoder;}
  TEncSlice* getSliceEncoder() {return m_pcSliceEncoder;}
  Void initLumaDeltaQpLUT();
  Int calculateLumaDQP(TComDataCU *pCU,
                       const UInt absPartIdx,
                       const TComYuv* pOrgYuv);
  Void create(UChar uhTotalDepth,
              UInt iMaxWidth,
              UInt iMaxHeight,
              ChromaFormat chromaFormat);
  Void destroy();
  Void compressCtu(TComDataCU* pCtu);
  Void encodeCtu(TComDataCU* pCtu,
                 const bool& isFilledMap);
  Int updateCtuDataISlice(TComDataCU* pCtu,
                          Int width,
                          Int height);
  Void setFastDeltaQp(Bool b){m_bFastDeltaQP = b;}

protected:
  Void finishCU(TComDataCU* pcCU,
                UInt uiAbsPartIdx);
#if AMP_ENC_SPEEDUP
  Void xCompressCU(TComDataCU*& rpcBestCU,
                   TComDataCU*& rpcTempCU,
                   const UInt uiDepth DEBUG_STRING_FN_DECLARE(sDebug),
                   PartSize eParentPartSize = NUMBER_OF_PART_SIZES);
#else
  Void xCompressCU(TComDataCU*& rpcBestCU,
                   TComDataCU*& rpcTempCU,
                   const UInt uiDepth);
#endif
  Void xEncodeCU(TComDataCU* pcCU,
                 UInt uiAbsPartIdx,
                 UInt uiDepth,
                 const bool& isFilledMap);
  Int xComputeQP(TComDataCU* pcCU,
                 UInt uiDepth);
  Void xCheckBestMode(TComDataCU*& rpcBestCU,
                      TComDataCU*& rpcTempCU,
                      UInt uiDepth DEBUG_STRING_FN_DECLARE(sParent) DEBUG_STRING_FN_DECLARE(sTest) DEBUG_STRING_PASS_INTO(Bool bAddSizeInfo=true));
  Void xCheckRDCostMerge2Nx2N(TComDataCU*& rpcBestCU,
                              TComDataCU*& rpcTempCU DEBUG_STRING_FN_DECLARE(sDebug),
                              Bool* earlyDetectionSkipMode);
#if AMP_MRG
  Void xCheckRDCostInter(TComDataCU*& rpcBestCU,
                         TComDataCU*& rpcTempCU,
                         PartSize ePartSize DEBUG_STRING_FN_DECLARE(sDebug),
                         Bool bUseMRG = false);
#else
  Void xCheckRDCostInter(TComDataCU*& rpcBestCU,
                         TComDataCU*& rpcTempCU,
                         PartSize ePartSize);
#endif
  Void xCheckRDCostIntra(TComDataCU*& rpcBestCU,
                         TComDataCU*& rpcTempCU,
                         PartSize ePartSize DEBUG_STRING_FN_DECLARE(sDebug));
  Void xCheckDQP(TComDataCU* pcCU);
  Void xCheckIntraPCM(TComDataCU*& rpcBestCU,
                      TComDataCU*& rpcTempCU);
  Void xCopyAMVPInfo(AMVPInfo* pSrc,
                     AMVPInfo* pDst);
  Void xCopyYuv2Pic(TComPic* rpcPic,
                    UInt uiCUAddr,
                    UInt uiAbsPartIdx,
                    UInt uiDepth,
                    UInt uiSrcDepth);
  Void xCopyYuv2Tmp(UInt uhPartUnitIdx,
                    UInt uiDepth);
  Bool getdQPFlag() {return m_bEncodeDQP;}
  Void setdQPFlag(Bool b) {m_bEncodeDQP = b;}
  Bool getFastDeltaQp() const {return m_bFastDeltaQP;}
  Bool getCodeChromaQpAdjFlag() {return m_stillToCodeChromaQpOffsetFlag;}
  Void setCodeChromaQpAdjFlag(Bool b) {m_stillToCodeChromaQpOffsetFlag = b;}
#if ADAPTIVE_QP_SELECTION
  Void xCtuCollectARLStats(TComDataCU* pCtu);
  Int  xTuCollectARLStats(TCoeff* rpcCoeff,
                          TCoeff* rpcArlCoeff,
                          Int NumCoeffInCU,
                          Double* cSum,
                          UInt* numSamples);
#endif
#if AMP_ENC_SPEEDUP
#if AMP_MRG
  Void deriveTestModeAMP(TComDataCU *pcBestCU,
                         PartSize eParentPartSize,
                         Bool& bTestAMP_Hor,
                         Bool& bTestAMP_Ver,
                         Bool& bTestMergeAMP_Hor,
                         Bool& bTestMergeAMP_Ver);
#else
  Void deriveTestModeAMP(TComDataCU* pcBestCU,
                         PartSize eParentPartSize,
                         Bool& bTestAMP_Hor,
                         Bool& bTestAMP_Ver);
#endif
#endif
  Void xFillPCMBuffer(TComDataCU* pCU,
                      TComYuv* pOrgYuv);
};

#endif
