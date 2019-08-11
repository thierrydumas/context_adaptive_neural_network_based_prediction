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

#ifndef __TENCSLICE__
#define __TENCSLICE__

#include "TLibCommon/CommonDef.h"
#include "TLibCommon/TComList.h"
#include "TLibCommon/TComPic.h"
#include "TLibCommon/TComPicYuv.h"
#include "TEncCu.h"
#include "WeightPredAnalysis.h"
#include "TEncRateCtrl.h"

class TEncTop;
class TEncGOP;

class TEncSlice : public WeightPredAnalysis
{
private:
  TEncCfg* m_pcCfg;
  TComList<TComPic*>* m_pcListPic;
  TComPicYuv m_picYuvPred;
  TComPicYuv m_picYuvResi;
  
  unsigned char* m_ptr_map_luminance;     /** < Pointer to the map of intra prediction modes for the luminance channel. */
  unsigned char* m_ptr_map_chrominance;   /** < Pointer to the map of intra prediction modes for the two chrominance channels. */
  
  TEncGOP* m_pcGOPEncoder;
  TEncCu* m_pcCuEncoder;
  TEncSearch* m_pcPredSearch;
  TEncEntropy* m_pcEntropyCoder;
  TEncSbac* m_pcSbacCoder;
  TEncBinCABAC* m_pcBinCABAC;
  TComTrQuant* m_pcTrQuant;
  TComRdCost* m_pcRdCost;
  TEncSbac*** m_pppcRDSbacCoder;
  TEncSbac* m_pcRDGoOnSbacCoder;
  UInt64 m_uiPicTotalBits;
  UInt64 m_uiPicDist;
  Double m_dPicRdCost;
  std::vector<Double> m_vdRdPicLambda;
  std::vector<Double> m_vdRdPicQp;
  std::vector<Int> m_viRdPicQp;
  TEncRateCtrl* m_pcRateCtrl;
  UInt m_uiSliceIdx;
  TEncSbac m_lastSliceSegmentEndContextState;
  TEncSbac m_entropyCodingSyncContextState;
  SliceType m_encCABACTableIdx;
  Int m_gopID;

  Double calculateLambda(const TComSlice* pSlice,
                         const Int GOPid,
                         const Int depth,
                         const Double refQP,
                         const Double dQP,
                         Int& iQP);
  Void setUpLambda(TComSlice* slice,
                   const Double dLambda,
                   Int iQP);
  Void calculateBoundingCtuTsAddrForSlice(UInt& startCtuTSAddrSlice,
                                          UInt& boundingCtuTSAddrSlice,
                                          Bool& haveReachedTileBoundary,
                                          TComPic* pcPic,
                                          const Int sliceMode,
                                          const Int sliceArgument);

public:
  TEncSlice();
  virtual ~TEncSlice();
  
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
  
  /** @brief Returns a pointer to the map of intra prediction modes for the luminance channel.
   *
   *  @return Pointer to the map of intra prediction modes for the
   *          luminance channel.
   */
  unsigned char* getPtrMapLuminance() const {return m_ptr_map_luminance;}
  
  /** @brief Returns a pointer to the map of intra prediction modes for the two chrominance channels.
   *
   *  @return Pointer to the map of intra prediction modes for the
   *          two chrominance channels.
   */
  unsigned char* getPtrMapChrominance() const {return m_ptr_map_chrominance;}

  Void create(Int iWidth,
              Int iHeight,
              ChromaFormat chromaFormat,
              UInt iMaxCUWidth,
              UInt iMaxCUHeight,
              UChar uhTotalDepth);
  Void destroy();
  Void init(TEncTop* pcEncTop);
  Void initEncSlice(TComPic* pcPic,
                    const Int pocLast,
                    const Int pocCurr,
                    const Int iGOPid,
                    TComSlice*& rpcSlice,
                    const Bool isField);
  Void resetQP(TComPic* pic,
               Int sliceQP,
               Double lambda);
  Void setGopID(Int iGopID) {m_gopID = iGopID;}
  Int getGopID() const {return m_gopID;}
  Void updateLambda(TComSlice* pSlice,
                    Double dQP);
  Void precompressSlice(TComPic* pcPic);
  Void compressSlice(TComPic* pcPic,
                     const Bool bCompressEntireSlice,
                     const Bool bFastDeltaQP);
  Void calCostSliceI(TComPic* pcPic);
  Void encodeSlice(TComPic* pcPic,
                   TComOutputBitstream* pcSubstreams,
                   UInt& numBinsCoded);
  Void setSearchRange(TComSlice* pcSlice);
  TEncCu* getCUEncoder() {return m_pcCuEncoder;}
  Void xDetermineStartAndBoundingCtuTsAddr(UInt& startCtuTsAddr,
                                           UInt& boundingCtuTsAddr,
                                           TComPic* pcPic);
  UInt getSliceIdx() {return m_uiSliceIdx;}
  Void setSliceIdx(UInt i) {m_uiSliceIdx = i;}
  SliceType getEncCABACTableIdx() const {return m_encCABACTableIdx;}
  
private:
  Double xGetQPValueAccordingToLambda(Double lambda);
};

#endif


