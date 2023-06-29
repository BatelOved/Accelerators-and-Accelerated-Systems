///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <memory>
#include "ex3.h"


/* Abstract base class for both parts of the exercise */
class image_processing_server
{
public:
    virtual ~image_processing_server() {}

    /* Enqueue a pair of images (target and reference) for processing. Receives pointers to pinned host
     * memory. Return false if there is no room for image (caller will try again).
     */
    virtual bool enqueue(int job_id, uchar *target, uchar *reference, uchar *result) = 0;

    /* Checks whether any pair of images has completed processing. If so, set job_id
     * accordingly, and return true. */
    virtual bool dequeue(int *job_id) = 0;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////

