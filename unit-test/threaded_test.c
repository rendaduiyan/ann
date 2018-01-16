/*
 * threaded_test.c, unit test for multiple-threaded bp-ann
 *
 * Copyright (c) 2018, haibolei <duiyanrenda@gmail.com>
 */

#include <stdlib.h>
#include "test.h"
#include "../comp_thread.h"

void init_weights (PlaneBin *pb)
{
    gdouble n = 0.0f;
    for (int i = 0; i < pb->m_w_num; i ++, n += 1.0f)
    {
        gdouble r = 0.1f;
        for (int j = 0; j < pb->m_dims[i].m_x; j ++, r += 0.1f)
        {
            gdouble c = 0.01f;
            for (int k = 0; k < pb->m_dims[i].m_y; k ++, c += 0.01f)
            {
                pb->m_weights[i][j][k] = n + r + c;
            }
        }
    }
}

void free_test (PlaneBin *pb)
{
    g_free (pb->m_in_data[0]);
    g_free (pb->m_in_data);
    g_free (pb->m_exp_val[0]);
    g_free (pb->m_exp_val);
}

void init_inputs (PlaneBin *pb)
{
    /*
    gdouble r = 0.1f;
    for (int i = 0; i < pb->m_size; i ++, r += 0.1f)
    {
        Layer *il = pb->m_plane[i]->m_layers[0];
        gdouble c = 0.01f;
        for (int j = 0; j < layer_raw_size (il); j ++, c += 0.01f)
        {
            layer_neuron (il, j)->m_output = r + c;
        }
    }
    */
    pb->m_n_iteration = 1;
    pb->m_n_sample = 1;
    pb->m_in_data = g_malloc (sizeof (gdouble *) * pb->m_n_sample);
    pb->m_exp_val = g_malloc (sizeof (gdouble *) * pb->m_n_sample);
    pb->m_in_data[0] =  g_malloc (sizeof (gdouble) * 2);
    pb->m_in_data[0][0] = 0.11f;
    pb->m_in_data[0][1] = 0.12f;
    pb->m_exp_val[0] = g_malloc (sizeof (gdouble));
    pb->m_exp_val[0][0] = 0.5f;
    pb->m_lr = 0.5f;
}

void test_all ()
{
    test_1_hidden ();
}


void check_1 (gpointer data)
{
    PlaneBin *pb = data;
    guint ln = 0;
    Layer *l = pb->m_plane[0]->m_layers[ln];

    g_assert_true (flt_equal (layer_neuron (l, 0)->m_output, 0.11f));
    g_assert_true (flt_equal (layer_neuron (l, 1)->m_output, 0.12f));
    g_assert_true (flt_equal (layer_neuron (l, 2)->m_output, 1.0f));
    /*
     * checking first hidden layer
     */
    ln = 1;
    l = pb->m_plane[0]->m_layers[ln];

    gdouble exp1 = 0.11f * 0.11f + 0.21f * 0.12f + 0.31f;
    g_assert_true (flt_equal (layer_neuron (l, 0)->m_output, exp1));
    gdouble exp2 = 0.12f * 0.11f + 0.22f * 0.12f + 0.32f;
    g_assert_true (flt_equal (layer_neuron (l, 1)->m_output, exp2));

    /*
     * checking output layer
     */
    ln ++;
    l = pb->m_plane[0]->m_layers[ln];
    gdouble exp3 = exp1 * 1.11f + exp2 * 1.21f + 1.31f;
    g_assert_true (flt_equal (layer_neuron (l, 0)->m_output, exp3));
    exp3 = layer_neuron (l, 0)->m_output;
}

void check_2 (gpointer data)
{
    PlaneBin *pb = data;
    guint ln = 2;
    Layer *l = pb->m_plane[0]->m_layers[ln];
    gdouble exp3 = layer_neuron (l, 0)->m_output;
    /*
     * checking backpropogation deltas
     */
    gdouble delta = (pb->m_exp_val[0][0] - exp3) * dact_relu (exp3);
    g_assert_true (flt_equal (layer_neuron (l, 0)->m_delta, delta));

    /*
     * last hidden layer
     */
    Layer *p = l;
    ln = 1;
    l = pb->m_plane[0]->m_layers[ln];
    gdouble o = layer_neuron (l, 0)->m_output;
    delta = 1.11f * layer_neuron (p, 0)->m_delta * dact_relu (o);
    g_assert_true (flt_equal (layer_neuron (l, 0)->m_delta, delta));

    o = layer_neuron (l, 1)->m_output;
    delta = 1.21f * layer_neuron (p, 0)->m_delta * dact_relu (o);
    g_assert_true (flt_equal (layer_neuron (l, 1)->m_delta, delta));

    o = layer_neuron (l, 2)->m_output;
    delta = 1.31f * layer_neuron (p, 0)->m_delta * dact_relu (o);
    g_assert_true (flt_equal (layer_neuron (l, 2)->m_delta, delta));

    
    /*
     * checking weights for output layer
     */
    ln = 2;
    l = pb->m_plane[0]->m_layers[ln];
    p = pb->m_plane[0]->m_layers[ln - 1];

    gdouble w = 1.11f + pb->m_lr * layer_neuron (l, 0)->m_delta * layer_neuron (p, 0)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 0, 0), w));
    w = 1.21f + pb->m_lr * layer_neuron (l, 0)->m_delta * layer_neuron (p, 1)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 1, 0), w));
    w = 1.31f + pb->m_lr * layer_neuron (l, 0)->m_delta * layer_neuron (p, 2)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 2, 0), w));

    /*
     * checking weights for first neuron of last hidden layer 
     */
    ln --;
    l = pb->m_plane[0]->m_layers[ln];
    p = pb->m_plane[0]->m_layers[ln - 1];

    w = 0.11f + pb->m_lr * layer_neuron (l, 0)->m_delta * layer_neuron (p, 0)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 0, 0), w));
    w = 0.21f + pb->m_lr * layer_neuron (l, 0)->m_delta * layer_neuron (p, 1)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 1, 0), w));
    w = 0.31f + pb->m_lr * layer_neuron (l, 0)->m_delta * layer_neuron (p, 2)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 2, 0), w));

    /*
     * checking weights for second neuron of last hidden layer 
     */
    w = 0.12f + pb->m_lr * layer_neuron (l, 1)->m_delta * layer_neuron (p, 0)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 0, 1), w));
    w = 0.22f + pb->m_lr * layer_neuron (l, 1)->m_delta * layer_neuron (p, 1)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 1, 1), w));
    w = 0.32f + pb->m_lr * layer_neuron (l, 1)->m_delta * layer_neuron (p, 2)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 2, 1), w));

    /*
     * checking weights for input layer
     */
}

void test_1_hidden ()
{
    PlaneBin *pb = construct_plane_bin (1, 2, 2, 1, 1);
    set_plane_func (pb->m_plane[0], act_relu, dact_relu);

    g_assert_cmpint (pb->m_size, ==, 1);
    g_assert_cmpint (pb->m_plane[0]->m_ln, ==, 3);

    guint ln = 0;

    Layer *l = pb->m_plane[0]->m_layers[ln];
    g_assert_cmpint (layer_size (l), ==, 3);
    g_assert_cmpint (layer_raw_size (l), ==, 2);
    g_assert_cmpint (layer_input_scale (l), ==, 1);
    ln ++;
    l = pb->m_plane[0]->m_layers[ln];

    g_assert_cmpint (layer_size (l), ==, 3);
    g_assert_cmpint (layer_raw_size (l), ==, 2);
    g_assert_cmpint (layer_input_scale (l), ==, 3);
    ln ++;
    l = pb->m_plane[0]->m_layers[ln];

    g_assert_cmpint (layer_size (l), ==, 1);
    g_assert_cmpint (layer_raw_size (l), ==, 1);
    g_assert_cmpint (layer_input_scale (l), ==, 3);

    ln = 0; //input layer
    l = pb->m_plane[0]->m_layers[ln];
    int i = 0;
    for (i = 0; i < layer_raw_size (l); i ++)
    {
        g_assert_nonnull (layer_neuron (l, i));
    }
    g_assert_true (flt_equal (layer_neuron (l, i)->m_output, 1.0f));

    init_inputs (pb);

    init_weights (pb);
    g_assert_cmpint (pb->m_w_num, ==, 2);
    g_assert_cmpint (pb->m_dims[0].m_x, ==, 3);
    g_assert_cmpint (pb->m_dims[0].m_y, ==, 2);
    g_assert_cmpint (pb->m_dims[1].m_x, ==, 3);
    g_assert_cmpint (pb->m_dims[1].m_y, ==, 1);

    pb->m_cbs.m_check_fwd_cb = check_1;
    pb->m_cbs.m_check_bwd_cb = check_2;
    

    ln ++; //first hidden layer
    l = pb->m_plane[0]->m_layers[ln];

    g_assert_true (*layer_neuron_w (l, 0, 0) == &pb->m_weights[0][0][0]);
    g_assert_true (*layer_neuron_w (l, 0, 1) == &pb->m_weights[0][1][0]);
    g_assert_true (*layer_neuron_w (l, 0, 2) == &pb->m_weights[0][2][0]);
    g_assert_true (flt_equal (layer_neuron_weight (l, 0, 0), 0.11f));
    g_assert_true (flt_equal (layer_neuron_weight (l, 0, 1), 0.21f));
    g_assert_true (flt_equal (layer_neuron_weight (l, 0, 2), 0.31f));

    g_assert_true (*layer_neuron_w (l, 1, 0) == &pb->m_weights[0][0][1]);
    g_assert_true (*layer_neuron_w (l, 1, 1) == &pb->m_weights[0][1][1]);
    g_assert_true (*layer_neuron_w (l, 1, 2) == &pb->m_weights[0][2][1]);
    g_assert_true (flt_equal (layer_neuron_weight (l, 1, 0), 0.12f));
    g_assert_true (flt_equal (layer_neuron_weight (l, 1, 1), 0.22f));
    g_assert_true (flt_equal (layer_neuron_weight (l, 1, 2), 0.32f));
    
    ln ++; //output layer
    l = pb->m_plane[0]->m_layers[ln];
    g_assert_true (*layer_neuron_w (l, 0, 0) == &pb->m_weights[ln - 1][0][0]);
    g_assert_true (*layer_neuron_w (l, 0, 1) == &pb->m_weights[ln - 1][1][0]);
    g_assert_true (*layer_neuron_w (l, 0, 2) == &pb->m_weights[ln - 1][2][0]);
    g_assert_true (flt_equal (layer_neuron_weight (l, 0, 0), 1.11f));
    g_assert_true (flt_equal (layer_neuron_weight (l, 0, 1), 1.21f));
    g_assert_true (flt_equal (layer_neuron_weight (l, 0, 2), 1.31f));


    init_threading (pb, 4);
    start_threading (pb);
    producer_main (pb);

    dump_plane_bin (pb, mask_da);
    free_plane_bin (pb);
    free_test (pb);
}

void test_2_hidden ()
{
}

int main (int argc, char **argv)
{
    g_test_init (&argc, &argv, NULL);
    g_test_add_func ("/ann/test", test_all);
    return g_test_run ();
}
