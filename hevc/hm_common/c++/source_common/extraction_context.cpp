#include "extraction_context.h"

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
                             const float& meanTraining)
{
    if (!piRoiOrigin)
    {
        fprintf(stderr, "`piRoiOrigin` is NULL.\n");
        return -1;
    }
    if (!piPortionAbove)
    {
        fprintf(stderr, "`piPortionAbove` is NULL.\n");
        return -1;
    }
    if (!piPortionLeft)
    {
        fprintf(stderr, "`piPortionLeft` is NULL.\n");
        return -1;
    }
    if (!bNeighborFlags)
    {
        fprintf(stderr, "`bNeighborFlags` is NULL.\n");
        return -1;
    }
    
    /*
    If `extract_context_portions` is called, the
    context is available. This implies that the
    current TB has at least one neighbouring unit.
    */
    if (iNumIntraNeighbor <= 0)
    {
        fprintf(stderr, "`iNumIntraNeighbor` is not strictly positive.\n");
        return -1;
    }
    const int* piTemp(NULL);
    const int iTotalUnits(iAboveUnits + iLeftUnits + 1);
    const int contextWidth(3*uiTuWidth);
    int i(0);
    int j(0);
    int k(0);

    // All the neighbouring units are available.
    if (iNumIntraNeighbor == iTotalUnits)
    {
        /*
        Here, `piTemp` points to the pixel at the top-left corner
        of the masked context portion located above the current TB
        in the already encoded and decoded channel.
        */
        piTemp = piRoiOrigin - uiTuHeight*iPicStride - uiTuWidth;
        for (i = 0; i < uiTuHeight; i++)
        {
            for (j = 0; j < contextWidth; j++)
            {
                piPortionAbove[i*contextWidth + j] = static_cast<float>(piTemp[j]) - meanTraining;
            }
            piTemp += iPicStride;
        }

        /*
        Now, `piTemp` points to the pixel in the already
        encoded and decoded channel whose row is the row
        of the pixel at the top-left corner of the current
        TB and whose column is the column of the pixel at
        the top-left corner of the masked context portion
        located above the current TB.
        */
        piTemp = piRoiOrigin - uiTuWidth;
        for (i = 0; i < 2*uiTuHeight; i++)
        {
            for (j = 0; j < uiTuWidth; j++)
            {
                piPortionLeft[i*uiTuWidth + j] = static_cast<float>(piTemp[j]) - meanTraining;
            }
            piTemp += iPicStride;
        }
    }

    // The neighbouring units are partially available.
    else
    {
        for (i = 0; i < uiTuHeight*contextWidth; i++)
        {
            piPortionAbove[i] = 0.;
        }
        for (i = 0; i < 2*uiTuHeight*uiTuWidth; i++)
        {
            piPortionLeft[i] = 0.;
        }

        /*
        Here, `piTemp` points to the pixel at the top-left corner
        of the masked context portion located above the current TB
        in the already encoded and decoded channel.
        */
        piTemp = piRoiOrigin - uiTuHeight*iPicStride - uiTuWidth;
        float* piPortionAboveTemp(piPortionAbove);

        /*
        The pixels in the already encoded and decoded channel
        belonging to Portion (1) are put into the buffer of the
        masked context portion located above the current TB. See
        the note "Note on the function `extract_context_portions`"
        for more details on Portion (1).
        */
        for (i = 0; i < uiTuHeight; i++)
        {
            for (j = 0; j < uiTuWidth; j++)
            {
                piPortionAboveTemp[j] = static_cast<float>(piTemp[j]) - meanTraining;
            }
            piPortionAboveTemp += contextWidth;
            piTemp += iPicStride;
        }
        
        /*
        The neighbouring unit above and on the left side of the
        current TB is necessarily available.
        */
        const bool* pbNeighborFlags(bNeighborFlags + iLeftUnits);
        if (!(*pbNeighborFlags))
        {
            fprintf(stderr, "The neighbouring unit above and on the left side of the current TB is not available.\n");
            return -1;
        }
        pbNeighborFlags++;
        
        /*
        The pixels in the already encoded and decoded channel
        belonging to Portion (a.1), Portion (a.2), ..., Portion (a.n)
        are put into the buffer of the masked context portion located
        above the current TB. See the note "Note on the function
        `extract_context_portions`" for more details on Portion
        (a.1), Portion (a.2), ..., Portion (a.n).
        */
        for (i = 0; i < iAboveUnits; i++)
        {
            if (*pbNeighborFlags)
            {
                piPortionAboveTemp = piPortionAbove + uiTuWidth + i*unitWidth;
                piTemp = piRoiOrigin - uiTuHeight*iPicStride + i*unitWidth;
                for (j = 0; j < uiTuHeight; j++)
                {
                    for (k = 0; k < unitWidth; k++)
                    {
                        piPortionAboveTemp[k] = static_cast<float>(piTemp[k]) - meanTraining;
                    }
                    piPortionAboveTemp += contextWidth;
                    piTemp += iPicStride;
                }
            }
            pbNeighborFlags++;
        }
        
        float* piPortionLeftTemp(piPortionLeft);
        
        /*
        Now, `piTemp` points to the pixel in the already
        encoded and decoded channel whose row is the row
        of the pixel at the top-left corner of the current
        TB and whose column is the column of the pixel at
        the top-left corner of the masked context portion
        located above the current TB.
        */
        piTemp = piRoiOrigin - uiTuWidth;
        
        /*
        The pixels in the already encoded and decoded channel
        belonging to Portion (l.1), Portion (l.2), ..., Portion (l.n)
        are put into the buffer of the masked context portion
        located on the left side of the current TB. See the
        note "Note on the function `extract_context_portions`"
        for more details on Portion (l.1), Portion (l.2), ...,
        Portion (l.n).
        */
        pbNeighborFlags = bNeighborFlags + iLeftUnits - 1;
        for (i = 0; i < iLeftUnits; i++)
        {
            if (*pbNeighborFlags)
            {
                for (j = 0; j < unitHeight; j++)
                {
                    for (k = 0; k < uiTuWidth; k++)
                    {
                        piPortionLeftTemp[k] = static_cast<float>(piTemp[k]) - meanTraining;
                    }
                    piPortionLeftTemp += uiTuWidth;
                    piTemp += iPicStride;
                }
            }
            pbNeighborFlags--;
        }
    }
    return 0;
}


