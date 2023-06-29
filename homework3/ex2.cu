/* This file should be almost identical to ex2.cu from homework 2. */
/* once the TODOs in this file are complete, the RPC version of the server/client should work correctly. */

#include "ex3.h"
#include "ex2.h"
#include <cuda/atomic>


__device__
void process_image(uchar *target, uchar *reference, uchar *result) {
    //TODO complete according to HW2
    //(This file should be almost identical to ex2.cu from homework 2.)
}


__global__
void process_image_kernel(uchar *target, uchar *reference, uchar *result){
    process_image(target, reference, result);
}


// TODO complete according to HW2:
//          implement a SPSC queue,
//          implement the persistent kernel,
//          implement a function for calculating the threadblocks count
// (This file should be almost identical to ex2.cu from homework 2.)


class queue_server : public image_processing_server
{
public:
    //TODO complete according to HW2
    //(This file should be almost identical to ex2.cu from homework 2.)

    queue_server(int threads)
    {
        //TODO complete according to HW2
        //(This file should be almost identical to ex2.cu from homework 2.)
    }

    ~queue_server() override
    {
        //TODO complete according to HW2
        //(This file should be almost identical to ex2.cu from homework 2.)
    }

    bool enqueue(int job_id, uchar *target, uchar *reference, uchar *result) override
    {
        //TODO complete according to HW2
        //(This file should be almost identical to ex2.cu from homework 2.)
        return false;
    }

    bool dequeue(int *job_id) override
    {
        //TODO complete according to HW2
        //(This file should be almost identical to ex2.cu from homework 2.)
        return false;
    }
};


std::unique_ptr<queue_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
