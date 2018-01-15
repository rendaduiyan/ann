/*
 * threaded_func_test.c, using logic and to test basic funcationalities 
 * of ANN with Node Parallelization.
 * Copyright (c) 2018, haibolei <duiyanrenda@gmail.com>
 */

#include <stdlib.h>
#include "func_test.h"

/*
 * TO-DO: implementation callbacks for threaded ANN
 */

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

void logic_and_test (gpointer data)
{
    /*
     * test the training result
     */

    PlaneBin *bin = data;
    gsize sample_num = 0;
    g_print ("testing the results for input samples:\n");
    SEP;
    //dump_plane_bin (bin, mask_dw | mask_dd);
    guint i = 0;
    p2d2 sample = init_inputs (bin, &sample_num);
    load_data (bin, (gdouble *)sample);
    fwd_plane_bin_act (bin);
    g_print ("output for test case %u is %g\n", i ++, bin->m_plane[0]->m_layers[2]->m_neurons[0]->m_output);
    ++ sample;
    load_data (bin, (gdouble *)sample);
    fwd_plane_bin_act (bin);
    g_print ("output for test case %u is %g\n", i ++, bin->m_plane[0]->m_layers[2]->m_neurons[0]->m_output);
    ++ sample;
    load_data (bin, (gdouble *)sample);
    fwd_plane_bin_act (bin);
    g_print ("output for test case %u is %g\n", i ++, bin->m_plane[0]->m_layers[2]->m_neurons[0]->m_output);
    ++ sample;
    load_data (bin, (gdouble *)sample);
    fwd_plane_bin_act (bin);
    g_print ("output for test case %u is %g\n", i ++, bin->m_plane[0]->m_layers[2]->m_neurons[0]->m_output);
    SEP;
}

int main (int argc, char **argv)
{
    guint plane_thread_num = 4;
    if (argc == 2)
    {
        plane_thread_num = atoi (argv[1]);
    }


    static const gdouble learning_rate = 0.8f;
    static const gsize training_iter = 500;
    static const guint in = 2;
    static const guint hn = 200;
    static const guint on = 1;
    static const guint hln = 1;

    PlaneBin *bin = construct_plane_bin (1, in, hn, on, hln);
    bin->m_cbs.m_test_cb = logic_and_test;
    //set_plane_func (bin->m_plane[0], act_relu, dact_relu);
    //set_plane_func (bin->m_plane[0], act_threshold, dact_threshold);
    bin->m_n_iteration = training_iter;
    bin->m_curr_iter = 0;
    bin->m_lr = learning_rate;
    gsize sample_num = 0;
    init_weights (bin);
    p2d2 sample = init_inputs (bin, &sample_num);
    gdouble *exp = init_expects (bin);
    bin->m_in_data = g_malloc (sizeof (gdouble *) * sample_num);
    bin->m_exp_val = g_malloc (sizeof (gdouble *) * sample_num);
    for (int i = 0; i < sample_num; i ++, ++ sample, ++ exp)
    {
        //bin->m_in_data[i] = g_malloc (sizeof (gdouble) * in);
        bin->m_exp_val[i] = g_malloc (sizeof (gdouble) * on);
        bin->m_in_data[i] = (gdouble*) sample;
        /*
        for (int j = 0; j < in; j ++)
        {
            bin->m_in_data[i][j] = *sample[j];
        }
        */
        for (int j = 0; j < on; j ++)
        {
            bin->m_exp_val[i][j] = exp[j];
        }
    }

    init_threading (bin, plane_thread_num);
    bin->m_n_sample = sample_num;
    gint64 start_time = g_get_real_time () / 1000;
    /*
     * trigger training
     */
    start_threading (bin);
    producer_main (bin);
    gint64 end_time = g_get_real_time () / 1000;
    g_print ("it cost %03lld (ms) to train %u time(s) with %u neuron(s) in hidden layer\n", (end_time - start_time), training_iter, hn);

    for (int i = 0; i < sample_num; i ++)
    {
        //g_free (bin->m_in_data[i]);
        g_free (bin->m_exp_val[i]);
    }
    g_free (bin->m_in_data);
    g_free (bin->m_exp_val);

    free_plane_bin (bin);

    return 0;
}
