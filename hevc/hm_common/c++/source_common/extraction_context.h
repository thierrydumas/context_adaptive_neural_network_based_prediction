#ifndef EXTRACTION_CONTEXT_H
#define EXTRACTION_CONTEXT_H

// The inclusion of "cstddef" defines NULL.
#include <iostream>
#include <cstddef>

/** @brief Extract two masked context portions from the already encoded and decoded channel.
  *
  * @param piRoiOrigin Pointer to the pixel located at the top-left
  *                    corner of the current TB in the already encoded
  *                    and decoded channel.
  * @param piPortionAbove Pointer to the buffer of the masked context portion
  *                       located above the current TB.
  * @param piPortionLeft Pointer to the buffer of the masked context portion
  *                      located on the left side of the current TB.
  * @param bNeighborFlags Pointer to the neighbouring unit flags buffer.
  *                       An element of this buffer is the flag signalizing
  *                       whether a neighbouring unit is available or not.
  * @param iNumIntraNeighbor Number of available neighbouring units. `iNumIntraNeighbor`
  *                          is strictly positive.
  * @param unitWidth Unit width. For the luminance channel, `unitWidth` is
  *                  equal to 4 pixels. For the two the chrominance channels,
  *                  `unitWidth` is equal to 2 pixels.
  * @param unitHeight Unit height.
  * @param iAboveUnits Number of units above the current TB.
  * @param iLeftUnits Number of units on the left side of the current TB.
  * @param uiTuWidth Width of the current TB in pixels.
  * @param uiTuHeight Height of the current TB in pixels.
  * @param iPicStride Width of the channel plus some padding in pixels.
  * @param meanTraining Mean pixels luminance computed over different
  *                     luminance images.
  * @return Error code. It is equal to -1 if an error occurs. It is equal to 0 otherwise.
  *
  */
int extract_context_portions(const int* const piRoiOrigin,
                             float* const piPortionAbove,
                             float* const piPortionLeft,
                             const bool* const bNeighborFlags,
                             const int& iNumIntraNeighbor,
                             const int& unitWidth,
                             const int& unitHeight,
                             const int& iAboveUnits,
                             const int& iLeftUnits,
                             const int& uiTuWidth,
                             const int& uiTuHeight,
                             const int& iPicStride,
                             const float& meanTraining);

#endif


