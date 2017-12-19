#include <stdlib.h>
#include "func_test.h"

void init_weights (PlaneBin *pb)
{
    init_weights_randomly (pb);
}

gdouble *init_expects (PlaneBin *pb)
{
    static gdouble outputs [] = 
    {
        1.0f, 0.0f, 0.0f, 0.0f
    };
    return outputs;
}

p2d2 init_inputs (PlaneBin *pb, gsize *n)
{
    static gdouble inputs[4][2] = 
    { 
        {1.0f, 1.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {0.0f, 0.0f}
    };
    static p2d2 input = inputs;
    *n = 4;
    return input;
}

int main (int argc, char **argv)
{
    guint plane_thread_num = 32;
    guint node_thread_num = 16;
    if (argc == 3)
    {
        plane_thread_num = atoi (argv[1]);
        node_thread_num = atoi (argv[2]);
    }


    static const gdouble learning_rate = 0.8f;
    static const gsize training_iter = 500;

    //PlaneBin *bin = construct_plane_bin (1, 2, 2, 1, 1);
    PlaneBin *bin = construct_plane_bin (1, 2, 200, 1, 1);
    //set_plane_func (bin->m_plane[0], act_relu, dact_relu);
    //set_plane_func (bin->m_plane[0], act_threshold, dact_threshold);
    gsize sample_num = 0;
    init_weights (bin);
    for (int i = 0; i < training_iter; i ++)
    {
        p2d2 sample = init_inputs (bin, &sample_num);
        gdouble *exp = init_expects (bin);
        for (int j = 0; j < sample_num; j ++, ++ sample, ++exp)
        {
            load_data (bin, (gdouble *) sample);
            fwd_plane_bin_act (bin);
            bp_plane_bin_act (bin, exp, learning_rate);
            dump_plane_bin (bin, mask_dd);
        }
    }
    /*
     * test the training result
     */

    DBG ("testing the results for input samples:\n");
    SEP;
    //dump_plane_bin (bin, mask_dw | mask_dd);
    p2d2 sample = init_inputs (bin, &sample_num);
    load_data (bin, (gdouble *)sample);
    fwd_plane_bin_act (bin);
    dump_plane_bin (bin, mask_do);
    ++ sample;
    load_data (bin, (gdouble *)sample);
    fwd_plane_bin_act (bin);
    dump_plane_bin (bin, mask_do);
    ++ sample;
    load_data (bin, (gdouble *)sample);
    fwd_plane_bin_act (bin);
    dump_plane_bin (bin, mask_do);
    ++ sample;
    load_data (bin, (gdouble *)sample);
    fwd_plane_bin_act (bin);
    dump_plane_bin (bin, mask_do);
    SEP;

    free_plane_bin (bin);

    return 0;
}
