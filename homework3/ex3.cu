/* CUDA 10.2 has a bug that prevents including <cuda/atomic> from two separate
 * object files. As a workaround, we include ex2.cu directly here. */
#include "ex2.cu"

#include <cassert>
#include <vector>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <infiniband/verbs.h>

#define RDMA_WC_CHECK(cq) do { \
    struct ibv_wc wc; \
    int ncqes; \
    while ((ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) {} \
    if (ncqes < 0) {\
        perror("ibv_poll_cq() failed"); \
        exit(1); \
    } \
    VERBS_WC_CHECK(wc); \
} while (0)

class server_rpc_context : public rdma_server_context {
private:
    std::unique_ptr<queue_server> gpu_context;
    std::array<int, OUTSTANDING_REQUESTS> input_read_done_count;


public:
    explicit server_rpc_context(uint16_t tcp_port) : rdma_server_context(tcp_port),
        gpu_context(create_queues_server(256))
    {
        std::fill(input_read_done_count.begin(), input_read_done_count.end(), 0);

    }

    virtual void event_loop() override
    {
        /* so the protocol goes like this:
         * 1. we'll wait for a CQE indicating that we got an Send request from the client.
         *    this tells us we have new work to do. The wr_id we used in post_recv tells us
         *    where the request is.
         * 2. now we send an RDMA Read to the client to retrieve the request.
         *    we will get a completion indicating the read has completed.
         * 3. we process the request on the GPU.
         * 4. upon completion, we send an RDMA Write with immediate to the client with
         *    the results.
         */
        rpc_request* req;
        uchar *img_target, *img_reference;
        uchar *img_out;

        bool terminate = false, got_last_cqe = false;

        while (!terminate || !got_last_cqe) {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
		VERBS_WC_CHECK(wc);

                switch (wc.opcode) {
                case IBV_WC_RECV:
                    /* Received a new request from the client */
                    req = &requests[wc.wr_id];
                    img_target = &images_target[wc.wr_id * IMG_BYTES];
                    img_reference = &images_reference[wc.wr_id * IMG_BYTES];
                    /* Terminate signal */
                    if (req->request_id == -1) {
                        printf("Terminating...\n");
                        terminate = true;
                        goto send_rdma_write;
                    }

                    /* Step 2: send RDMA Read to client to read the input */
                    input_read_done_count[wc.wr_id] = 0;
                    post_rdma_read(
                        img_target,                 // local_src
                        req->input_target_length,   // len
                        mr_images_target->lkey,     // lkey
                        req->input_target_addr,     // remote_dst
                        req->input_target_rkey,     // rkey
                        wc.wr_id);                  // wr_id

                    post_rdma_read(
                        img_reference,              // local_src
                        req->input_reference_length,// len
                        mr_images_reference->lkey,  // lkey
                        req->input_reference_addr,  // remote_dst
                        req->input_reference_rkey,  // rkey
                        wc.wr_id);                  // wr_id
                break;

                case IBV_WC_RDMA_READ:
                    /* Completed RDMA read for a request */
                    input_read_done_count[wc.wr_id]++;
                    if (input_read_done_count[wc.wr_id] == 2){
                        req = &requests[wc.wr_id];
                        img_target = &images_target[wc.wr_id * IMG_BYTES];
                        img_reference = &images_reference[wc.wr_id * IMG_BYTES];
                        img_out = &images_out[wc.wr_id * IMG_BYTES];

                        // Step 3: Process on GPU
                        while(!gpu_context->enqueue(wc.wr_id, img_target, img_reference, img_out)){};
                    }
		        break;
                    
                case IBV_WC_RDMA_WRITE:
                    /* Completed RDMA Write - reuse buffers for receiving the next requests */
                    post_recv(wc.wr_id % OUTSTANDING_REQUESTS);

                    if (terminate)
                    got_last_cqe = true;

                break;
                default:
                    printf("Unexpected completion\n");
                    assert(false);
                }
            }

            // Dequeue completed GPU tasks
            int dequeued_job_id;
            if (gpu_context->dequeue(&dequeued_job_id)) {
                req = &requests[dequeued_job_id];
                img_out = &images_out[dequeued_job_id * IMG_BYTES];

send_rdma_write:
                // Step 4: Send RDMA Write with immediate to client with the response
		post_rdma_write(
                    req->output_addr,                       // remote_dst
                    terminate ? 0 : req->output_length,     // len
                    req->output_rkey,                       // rkey
                    terminate ? 0 : img_out,                // local_src
                    mr_images_out->lkey,                    // lkey
                    dequeued_job_id + OUTSTANDING_REQUESTS, // wr_id
                    (uint32_t*)&req->request_id);           // immediate
            }
        }
    }
};

class client_rpc_context : public rdma_client_context {
private:
    uint32_t requests_sent = 0;
    uint32_t send_cqes_received = 0;

    struct ibv_mr *mr_images_target, *mr_images_reference; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
public:
    explicit client_rpc_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
    }

    ~client_rpc_context()
    {
        kill();
    }

    virtual void set_input_images(uchar *images_target, uchar* images_reference, size_t bytes) override
    {
        /* register a memory region for the input images. */
        mr_images_target = ibv_reg_mr(pd, images_target, bytes, IBV_ACCESS_REMOTE_READ);
        if (!mr_images_target) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
        mr_images_reference = ibv_reg_mr(pd, images_reference, bytes, IBV_ACCESS_REMOTE_READ);
        if (!mr_images_reference) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

    virtual bool enqueue(int job_id, uchar *target, uchar *reference, uchar *result) override
    {
        if ((requests_sent - send_cqes_received) == OUTSTANDING_REQUESTS)
            return false;

        struct ibv_sge sg; /* scatter/gather element */
        struct ibv_send_wr wr; /* WQE */
        struct ibv_send_wr *bad_wr; /* ibv_post_send() reports bad WQEs here */

        /* step 1: send request to server using Send operation */
        
        struct rpc_request *req = &requests[requests_sent % OUTSTANDING_REQUESTS];
        req->request_id = job_id;
        req->input_target_rkey = target ? mr_images_target->rkey : 0;
        req->input_target_addr = (uintptr_t)target;
        req->input_target_length = IMG_BYTES;
        req->input_reference_rkey = reference ? mr_images_reference->rkey : 0;
        req->input_reference_addr = (uintptr_t)reference;
        req->input_reference_length = IMG_BYTES;
        req->output_rkey = result ? mr_images_out->rkey : 0;
        req->output_addr = (uintptr_t)result;
        req->output_length = IMG_BYTES;

        /* RDMA send needs a gather element (local buffer)*/
        memset(&sg, 0, sizeof(struct ibv_sge));
        sg.addr = (uintptr_t)req;
        sg.length = sizeof(*req);
        sg.lkey = mr_requests->lkey;

        /* WQE */
        memset(&wr, 0, sizeof(struct ibv_send_wr));
        wr.wr_id = job_id; /* helps identify the WQE */
        wr.sg_list = &sg;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED; /* always set this in this excersize. generates CQE */

        /* post the WQE to the HCA to execute it */
        if (ibv_post_send(qp, &wr, &bad_wr)) {
            perror("ibv_post_send() failed");
            exit(1);
        }

        ++requests_sent;

        return true;
    }

    virtual bool dequeue(int *job_id) override
    {
        /* When WQE is completed we expect a CQE */
        /* We also expect a completion of the RDMA Write with immediate operation from the server to us */
        /* The order between the two is not guarenteed */

        struct ibv_wc wc; /* CQE */
        int ncqes = ibv_poll_cq(cq, 1, &wc);
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        if (ncqes == 0)
            return false;

	VERBS_WC_CHECK(wc);

        switch (wc.opcode) {
        case IBV_WC_SEND:
            ++send_cqes_received;
            return false;
        case IBV_WC_RECV_RDMA_WITH_IMM:
            *job_id = wc.imm_data;
            break;
        default:
            printf("Unexpected completion type\n");
            assert(0);
        }

        /* step 2: post receive buffer for the next RPC call (next RDMA write with imm) */
        post_recv();

        return true;
    }

    void kill()
    {
        while (!enqueue(-1, // Indicate termination
                       NULL, NULL, NULL)) ;
        int job_id = 0;
        bool dequeued;
        do {
            dequeued = dequeue(&job_id);
        } while (!dequeued || job_id != -1);
    }
};

struct queues_exchange_data {
    int number_of_queues;

    uint32_t mr_cpu_to_gpu_rkey, mr_gpu_to_cpu_rkey;
    SPSC *mr_cpu_to_gpu_addr, *mr_gpu_to_cpu_addr;

    uint32_t mr_images_target_rkey, mr_images_reference_rkey, mr_images_out_rkey;
    uchar *mr_images_target_addr, *mr_images_reference_addr, *mr_images_out_addr;
};

class server_queues_context : public rdma_server_context {
private:
    std::unique_ptr<queue_server> gpu_context;
    bool running = false;

    /* Memory regions for CPU-GPU queues */
    struct ibv_mr* mr_cpu_to_gpu;
    struct ibv_mr* mr_gpu_to_cpu;

public:
    explicit server_queues_context(uint16_t tcp_port) :
        rdma_server_context(tcp_port),
        gpu_context(create_queues_server(256)) {
        /* Initialize additional server MRs as needed. */
        SPSC* cpu_to_gpu_queues = gpu_context->cpu_to_gpu_queues;
        SPSC* gpu_to_cpu_queues = gpu_context->gpu_to_cpu_queues;
        int blocks = gpu_context->blocks;

        mr_cpu_to_gpu = ibv_reg_mr(pd, cpu_to_gpu_queues, blocks * sizeof(SPSC), IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_cpu_to_gpu) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
        mr_gpu_to_cpu = ibv_reg_mr(pd, gpu_to_cpu_queues, blocks * sizeof(SPSC), IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_gpu_to_cpu) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }

        /* Exchange rkeys, addresses, and necessary information (e.g.
         * number of queues) with the client */
        struct queues_exchange_data my_info = {};
        my_info.number_of_queues = blocks;
        my_info.mr_cpu_to_gpu_rkey = mr_cpu_to_gpu->rkey;
        my_info.mr_gpu_to_cpu_rkey = mr_gpu_to_cpu->rkey;

        my_info.mr_cpu_to_gpu_addr = (SPSC*)mr_cpu_to_gpu->addr;
        my_info.mr_gpu_to_cpu_addr = (SPSC*)mr_gpu_to_cpu->addr;

        my_info.mr_images_target_rkey = mr_images_target->rkey;
        my_info.mr_images_reference_rkey = mr_images_reference->rkey;
        my_info.mr_images_out_rkey = mr_images_out->rkey;

        my_info.mr_images_target_addr = (uchar*)mr_images_target->addr;
        my_info.mr_images_reference_addr = (uchar*)mr_images_reference->addr;
        my_info.mr_images_out_addr = (uchar*)mr_images_out->addr;

        send_over_socket(&my_info, sizeof(struct queues_exchange_data));
    }

    ~server_queues_context() {
        /* Destroy the additional server MRs here */
        ibv_dereg_mr(mr_cpu_to_gpu);
        ibv_dereg_mr(mr_gpu_to_cpu);
    }

    virtual void event_loop() override {
        /* The server tasks are:
            1) Initialize CUDA environment, allocate CPU buffers mapped for the GPU, and
            instantiate the CUDA kernel (taken from homework 2).
            2) Register the memory that the GPU accesses also for remote access through
            verbs.
            3) Establish a connection with the client and send the client all necessary
            information to access these buffers over TCP.
         */

        running = true;
        while (running == true) {
            recv_over_socket(&running, sizeof(bool));
        }
    }
};

class client_queues_context : public rdma_client_context {
private:
    /* Server's necessary communication data for RDMA operations */
    struct queues_exchange_data remote_info;

    /* Memory region for input and output images */
    struct ibv_mr *mr_images_target, *mr_images_reference; 
    struct ibv_mr *mr_images_out;

    /* Memory region for CPU-GPU queues */
    SPSC curr_cpu_to_gpu_queue, curr_gpu_to_cpu_queue;
    struct ibv_mr *mr_curr_cpu_to_gpu_queue, *mr_curr_gpu_to_cpu_queue;

    /* Memory region for input and output requests */
    job_context request_in, request_out;
    struct ibv_mr *mr_request_in, *mr_request_out;

    uint32_t requests_sent = 0;
    uint32_t send_cqes_received = 0;

public:
    client_queues_context(uint16_t tcp_port) : rdma_client_context(tcp_port) {
        /* Communicate with server to discover number of queues, necessary
         * rkeys / address, or other additional information needed to operate
         * the GPU queues remotely. */
        recv_over_socket(&remote_info, sizeof(queues_exchange_data));

        /* Register memory regions for CPU-GPU queues */
        mr_curr_cpu_to_gpu_queue = ibv_reg_mr(pd, &curr_cpu_to_gpu_queue, sizeof(SPSC), IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_curr_cpu_to_gpu_queue) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }

        mr_curr_gpu_to_cpu_queue = ibv_reg_mr(pd, &curr_gpu_to_cpu_queue, sizeof(SPSC), IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_curr_gpu_to_cpu_queue) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }

        mr_request_in = ibv_reg_mr(pd, &request_in, sizeof(job_context), IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_request_in) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }

        mr_request_out = ibv_reg_mr(pd, &request_out, sizeof(job_context), IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_request_out) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    ~client_queues_context() {
        /* Terminate the server and release memory regions and other resources */
        bool running = false;
        send_over_socket(&running, sizeof(bool));
        ibv_dereg_mr(mr_curr_cpu_to_gpu_queue);
        ibv_dereg_mr(mr_curr_gpu_to_cpu_queue);
        ibv_dereg_mr(mr_request_in);
        ibv_dereg_mr(mr_request_out);
        ibv_dereg_mr(mr_images_target);
        ibv_dereg_mr(mr_images_reference);
        ibv_dereg_mr(mr_images_out);
    }

    virtual void set_input_images(uchar* images_target, uchar* images_reference, size_t bytes) override {
        /* Register a memory region for the input images. */
        mr_images_target = ibv_reg_mr(pd, images_target, bytes, IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_target) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
        mr_images_reference = ibv_reg_mr(pd, images_reference, bytes, IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_reference) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar* images_out, size_t bytes) override {
        /* Register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual bool enqueue(int job_id, uchar* target, uchar* reference, uchar* result) override {
        /* Enqueue:
            1) Checks if there is room on the GPU queue to submit new tasks, using an RDMA read operation.
            2) If there is room, copy the input images to the server and enqueue the next
            task on the GPU’s queue using RDMA write operations.
        */

        if (requests_sent - send_cqes_received == OUTSTANDING_REQUESTS) {
            return false;
        }

        int queue_pi = requests_sent % remote_info.number_of_queues;
        uint64_t curr_cpu_to_gpu_queue_remote_addr = ((uint64_t)remote_info.mr_cpu_to_gpu_addr + (queue_pi * sizeof(SPSC)));
        uint64_t curr_image_target_remote_addr = (uint64_t)remote_info.mr_images_target_addr + ((job_id % OUTSTANDING_REQUESTS) * IMG_BYTES);
        uint64_t curr_image_reference_remote_addr = (uint64_t)remote_info.mr_images_reference_addr + ((job_id % OUTSTANDING_REQUESTS) * IMG_BYTES);
        uint64_t curr_image_out_remote_addr = (uint64_t)remote_info.mr_images_out_addr + ((job_id % OUTSTANDING_REQUESTS) * IMG_BYTES);

        post_rdma_read(
            &curr_cpu_to_gpu_queue,                         // local_src
            sizeof(SPSC),                                   // len
            mr_curr_cpu_to_gpu_queue->lkey,                 // lkey
            curr_cpu_to_gpu_queue_remote_addr,              // remote_dst
            remote_info.mr_cpu_to_gpu_rkey,                 // rkey
            job_id                                          // wr_id
        );

        RDMA_WC_CHECK(cq);
        
        if (curr_cpu_to_gpu_queue.is_full()) {
            return false;
        }

        post_rdma_write(
            curr_image_target_remote_addr,                  // remote_dst
            IMG_BYTES,                                      // len
            remote_info.mr_images_target_rkey,              // rkey
            target,                                         // local_src
            mr_images_target->lkey,                         // lkey
            job_id                                          // wr_id
        );

        RDMA_WC_CHECK(cq);

        post_rdma_write(
            curr_image_reference_remote_addr,               // remote_dst
            IMG_BYTES,                                      // len
            remote_info.mr_images_reference_rkey,           // rkey
            reference,                                      // local_src
            mr_images_reference->lkey,                      // lkey
            job_id                                          // wr_id
        );

        RDMA_WC_CHECK(cq);

        request_in.job_id = job_id;
        request_in.target = (uchar*)(curr_image_target_remote_addr);
        request_in.reference = (uchar*)(curr_image_reference_remote_addr);
        request_in.img_out = (uchar*)(curr_image_out_remote_addr);
        request_in.remote_img_out = (uchar*)(result);

        curr_cpu_to_gpu_queue.push(request_in);

        post_rdma_write(
            curr_cpu_to_gpu_queue_remote_addr,             // remote_dst
            sizeof(SPSC),                                   // len
            remote_info.mr_cpu_to_gpu_rkey,                 // rkey
            &curr_cpu_to_gpu_queue,                         // local_src
            mr_curr_cpu_to_gpu_queue->lkey,                 // lkey
            job_id                                          // wr_id
        );

        RDMA_WC_CHECK(cq);

        ++requests_sent;

        return true;
    }

    virtual bool dequeue(int* img_id) override {
        /* Dequeue:
            1) Polls the GPU queue using RDMA read operations for any tasks that have
            been completed. You may wait for the RDMA read operation to complete,
            but you must not wait for the GPU to complete each task before moving on
            to the next step.
            2) If a task has been completed, use RDMA read operations to read the job ID
            and image content back to the client machine.
        */

        int wr_id = 1;
        int queue_ci = send_cqes_received % remote_info.number_of_queues;
        uint64_t curr_gpu_to_cpu_queue_remote_addr = (uint64_t)remote_info.mr_gpu_to_cpu_addr + (queue_ci * sizeof(SPSC));

        post_rdma_read(
            &curr_gpu_to_cpu_queue,                         // local_src
            sizeof(SPSC),                                   // len
            mr_curr_gpu_to_cpu_queue->lkey,                 // lkey
            curr_gpu_to_cpu_queue_remote_addr,              // remote_dst
            remote_info.mr_gpu_to_cpu_rkey,                 // rkey
            wr_id                                           // wr_id
        );

        RDMA_WC_CHECK(cq);

        if (curr_gpu_to_cpu_queue.is_empty()) {
            return false;
        }

        curr_gpu_to_cpu_queue.pop(&request_out);

        post_rdma_read(
            (void*)request_out.remote_img_out,              // local_src
            IMG_BYTES,                                      // len
            mr_images_out->lkey,                            // lkey
            (uint64_t)request_out.img_out,                  // remote_dst
            remote_info.mr_images_out_rkey,                 // rkey
            wr_id                                           // wr_id
        );

        RDMA_WC_CHECK(cq);

        post_rdma_write(
            curr_gpu_to_cpu_queue_remote_addr,              // remote_dst
            sizeof(SPSC),                                   // len
            remote_info.mr_gpu_to_cpu_rkey,                 // rkey
            &curr_gpu_to_cpu_queue,                         // local_src
            mr_curr_gpu_to_cpu_queue->lkey,                 // lkey
            wr_id                                           // wr_id
        );

        RDMA_WC_CHECK(cq);

        *img_id = request_out.job_id;
        ++send_cqes_received;

        return true;
    }
};

std::unique_ptr<rdma_server_context> create_server(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<server_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<server_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}

std::unique_ptr<rdma_client_context> create_client(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<client_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<client_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}
