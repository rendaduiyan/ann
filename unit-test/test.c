#include <stdlib.h>
#include "test.h"

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

void init_inputs (PlaneBin *pb)
{
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
}
void test_all ()
{
    //test_1_hidden ();
    test_2_hidden ();
    test_data_loading ();
}

void test_data_loading ()
{
    static gdouble inputs[4][2] = 
    { 
        {0.1f, 0.2f},
        {0.3f, 0.4f}
    };
    PlaneBin *bin = construct_plane_bin (1, 2, 2, 1, 1);
    load_data (bin, inputs[0]);
    g_assert_true (flt_equal (bin->m_plane[0]->m_layers[0]->m_neurons[0]->m_output, 0.1f));
    g_assert_true (flt_equal (bin->m_plane[0]->m_layers[0]->m_neurons[1]->m_output, 0.2f));
    load_data (bin, inputs[1]);
    g_assert_true (flt_equal (bin->m_plane[0]->m_layers[0]->m_neurons[0]->m_output, 0.3f));
    g_assert_true (flt_equal (bin->m_plane[0]->m_layers[0]->m_neurons[1]->m_output, 0.4f));
    free_plane_bin (bin);
}

void test_2_hidden ()
{
    PlaneBin *bin = construct_plane_bin (1, 2, 2, 1, 2);
    set_plane_func (bin->m_plane[0], act_relu, dact_relu);

    g_assert_cmpint (bin->m_size, ==, 1);
    g_assert_cmpint (bin->m_plane[0]->m_ln, ==, 4);

    guint ln = 0;

    Layer *l = bin->m_plane[0]->m_layers[ln];
    g_assert_cmpint (layer_size (l), ==, 3);
    g_assert_cmpint (layer_raw_size (l), ==, 2);
    g_assert_cmpint (layer_input_scale (l), ==, 1);
    ln ++;
    l = bin->m_plane[0]->m_layers[ln];

    g_assert_cmpint (layer_size (l), ==, 3);
    g_assert_cmpint (layer_raw_size (l), ==, 2);
    g_assert_cmpint (layer_input_scale (l), ==, 3);
    ln ++;
    l = bin->m_plane[0]->m_layers[ln];

    g_assert_cmpint (layer_size (l), ==, 3);
    g_assert_cmpint (layer_raw_size (l), ==, 2);
    g_assert_cmpint (layer_input_scale (l), ==, 3);
    ln ++;
    l = bin->m_plane[0]->m_layers[ln];

    g_assert_cmpint (layer_size (l), ==, 1);
    g_assert_cmpint (layer_raw_size (l), ==, 1);
    g_assert_cmpint (layer_input_scale (l), ==, 3);

    ln = 0; //input layer
    l = bin->m_plane[0]->m_layers[ln];
    int i = 0;
    for (i = 0; i < layer_raw_size (l); i ++)
    {
        g_assert_nonnull (layer_neuron (l, i));
    }
    g_assert_true (flt_equal (layer_neuron (l, i)->m_output, 1.0f));

    init_inputs (bin);
    g_assert_true (flt_equal (layer_neuron (l, 0)->m_output, 0.11f));
    g_assert_true (flt_equal (layer_neuron (l, 1)->m_output, 0.12f));
    g_assert_true (flt_equal (layer_neuron (l, 2)->m_output, 1.0f));

    init_weights (bin);
    g_assert_cmpint (bin->m_w_num, ==, 3);
    g_assert_cmpint (bin->m_dims[0].m_x, ==, 3);
    g_assert_cmpint (bin->m_dims[0].m_y, ==, 2);
    g_assert_cmpint (bin->m_dims[1].m_x, ==, 3);
    g_assert_cmpint (bin->m_dims[1].m_y, ==, 2);
    g_assert_cmpint (bin->m_dims[2].m_x, ==, 3);
    g_assert_cmpint (bin->m_dims[2].m_y, ==, 1);

    ln ++; //first hidden layer
    l = bin->m_plane[0]->m_layers[ln];

    g_assert_true (*layer_neuron_w (l, 0, 0) == &bin->m_weights[0][0][0]);
    g_assert_true (*layer_neuron_w (l, 0, 1) == &bin->m_weights[0][1][0]);
    g_assert_true (*layer_neuron_w (l, 0, 2) == &bin->m_weights[0][2][0]);
    g_assert_true (flt_equal (layer_neuron_weight (l, 0, 0), 0.11f));
    g_assert_true (flt_equal (layer_neuron_weight (l, 0, 1), 0.21f));
    g_assert_true (flt_equal (layer_neuron_weight (l, 0, 2), 0.31f));

    g_assert_true (*layer_neuron_w (l, 1, 0) == &bin->m_weights[0][0][1]);
    g_assert_true (*layer_neuron_w (l, 1, 1) == &bin->m_weights[0][1][1]);
    g_assert_true (*layer_neuron_w (l, 1, 2) == &bin->m_weights[0][2][1]);
    g_assert_true (flt_equal (layer_neuron_weight (l, 1, 0), 0.12f));
    g_assert_true (flt_equal (layer_neuron_weight (l, 1, 1), 0.22f));
    g_assert_true (flt_equal (layer_neuron_weight (l, 1, 2), 0.32f));
    
    ln ++; //second hidden layer
    l = bin->m_plane[0]->m_layers[ln];
    g_assert_true (*layer_neuron_w (l, 0, 0) == &bin->m_weights[1][0][0]);
    g_assert_true (*layer_neuron_w (l, 0, 1) == &bin->m_weights[1][1][0]);
    g_assert_true (*layer_neuron_w (l, 0, 2) == &bin->m_weights[1][2][0]);
    g_assert_true (flt_equal (layer_neuron_weight (l, 0, 0), 1.11f));
    g_assert_true (flt_equal (layer_neuron_weight (l, 0, 1), 1.21f));
    g_assert_true (flt_equal (layer_neuron_weight (l, 0, 2), 1.31f));

    g_assert_true (*layer_neuron_w (l, 1, 0) == &bin->m_weights[1][0][1]);
    g_assert_true (*layer_neuron_w (l, 1, 1) == &bin->m_weights[1][1][1]);
    g_assert_true (*layer_neuron_w (l, 1, 2) == &bin->m_weights[1][2][1]);
    g_assert_true (flt_equal (layer_neuron_weight (l, 1, 0), 1.12f));
    g_assert_true (flt_equal (layer_neuron_weight (l, 1, 1), 1.22f));
    g_assert_true (flt_equal (layer_neuron_weight (l, 1, 2), 1.32f));

    ln ++; //output layer
    l = bin->m_plane[0]->m_layers[ln];
    g_assert_true (*layer_neuron_w (l, 0, 0) == &bin->m_weights[2][0][0]);
    g_assert_true (*layer_neuron_w (l, 0, 1) == &bin->m_weights[2][1][0]);
    g_assert_true (*layer_neuron_w (l, 0, 2) == &bin->m_weights[2][2][0]);
    g_assert_true (flt_equal (layer_neuron_weight (l, 0, 0), 2.11f));
    g_assert_true (flt_equal (layer_neuron_weight (l, 0, 1), 2.21f));
    g_assert_true (flt_equal (layer_neuron_weight (l, 0, 2), 2.31f));


    fwd_plane_bin_act (bin);
    /*
     * checking first hidden layer
     */

    ln = 1;
    l = bin->m_plane[0]->m_layers[ln];

    gdouble exp1 = 0.11f * 0.11f + 0.21f * 0.12f + 0.31;
    g_assert_true (flt_equal (layer_neuron (l, 0)->m_output, exp1));
    gdouble exp2 = 0.12f * 0.11f + 0.22f * 0.12f + 0.32f;
    g_assert_true (flt_equal (layer_neuron (l, 1)->m_output, exp2));

    /*
     * checking second hidden layer
     */
    ln = 2;
    l = bin->m_plane[0]->m_layers[ln];

    gdouble exp3 = exp1 * 1.11f + exp2 * 1.21f + 1.31f;
    g_assert_true (flt_equal (layer_neuron (l, 0)->m_output, exp3));
    gdouble exp4 = exp1 * 1.12f + exp2 * 1.22f + 1.32f;
    g_assert_true (flt_equal (layer_neuron (l, 1)->m_output, exp4));

    /*
     * checking output layer
     */
    ln = 3;
    l = bin->m_plane[0]->m_layers[ln];
    gdouble exp5 = exp3 * 2.11f + exp4 * 2.21f + 2.31f;
    g_assert_true (flt_equal (layer_neuron (l, 0)->m_output, exp5));
    exp5 = layer_neuron (l, 0)->m_output;

    /*
     * checking backpropogation deltas
     */
    gdouble exp_val[1] = {0.5f};
    gdouble learning_rate = 0.5f;
    bp_plane_bin_act (bin, exp_val, learning_rate);
    gdouble delta = (exp_val[0] - exp5) * dact_relu (exp5);
    g_assert_true (flt_equal (layer_neuron (l, 0)->m_delta, delta));

    /*
     * last hidden layer
     */
    Layer *p = l;
    ln = 2;
    l = bin->m_plane[0]->m_layers[ln];
    gdouble o = layer_neuron (l, 0)->m_output;
    delta = 2.11f * layer_neuron (p, 0)->m_delta * dact_relu (o);
    g_assert_true (flt_equal (layer_neuron (l, 0)->m_delta, delta));

    o = layer_neuron (l, 1)->m_output;
    delta = 2.21f * layer_neuron (p, 0)->m_delta * dact_relu (o);
    g_assert_true (flt_equal (layer_neuron (l, 1)->m_delta, delta));

    o = layer_neuron (l, 2)->m_output;
    delta = 2.31f * layer_neuron (p, 0)->m_delta * dact_relu (o);
    g_assert_true (flt_equal (layer_neuron (l, 2)->m_delta, delta));

    /*
     * first hidden layer
     */
    p = l;
    ln = 1;
    l = bin->m_plane[0]->m_layers[ln];

    o = layer_neuron (l, 0)->m_output;
    gdouble d = 1.11f * layer_neuron (p, 0)->m_delta;
    d += (1.12f * layer_neuron (p, 1)->m_delta);
    delta = dact_relu (o) * d;
    g_assert_true (flt_equal (layer_neuron (l, 0)->m_delta, delta));

    o = layer_neuron (l, 1)->m_output;
    d = 1.21f * layer_neuron (p, 0)->m_delta;
    d += (1.22f * layer_neuron (p, 1)->m_delta);
    delta = dact_relu (o) * d;
    g_assert_true (flt_equal (layer_neuron (l, 1)->m_delta, delta));
    
    /*
     * checking weights for output layer
     */
    ln = 3;
    l = bin->m_plane[0]->m_layers[ln];
    p = bin->m_plane[0]->m_layers[ln - 1];

    gdouble w = 2.11f + learning_rate * layer_neuron (l, 0)->m_delta * layer_neuron (p, 0)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 0, 0), w));
    w = 2.21f + learning_rate * layer_neuron (l, 0)->m_delta * layer_neuron (p, 1)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 1, 0), w));
    w = 2.31f + learning_rate * layer_neuron (l, 0)->m_delta * layer_neuron (p, 2)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 2, 0), w));

    /*
     * checking weights for first neuron of last hidden layer 
     */
    ln = 2;
    l = bin->m_plane[0]->m_layers[ln];
    p = bin->m_plane[0]->m_layers[ln - 1];

    w = 1.11f + learning_rate * layer_neuron (l, 0)->m_delta * layer_neuron (p, 0)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 0, 0), w));
    w = 1.21f + learning_rate * layer_neuron (l, 0)->m_delta * layer_neuron (p, 1)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 1, 0), w));
    w = 1.31f + learning_rate * layer_neuron (l, 0)->m_delta * layer_neuron (p, 2)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 2, 0), w));

    /*
     * checking weights for second neuron of last hidden layer 
     */
    w = 1.12f + learning_rate * layer_neuron (l, 1)->m_delta * layer_neuron (p, 0)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 0, 1), w));
    w = 1.22f + learning_rate * layer_neuron (l, 1)->m_delta * layer_neuron (p, 1)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 1, 1), w));
    w = 1.32f + learning_rate * layer_neuron (l, 1)->m_delta * layer_neuron (p, 2)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 2, 1), w));

    /*
     * checking weights for first hidden layer 
     */
    ln = 1;
    l = bin->m_plane[0]->m_layers[ln];
    p = bin->m_plane[0]->m_layers[ln - 1];

    w = 0.11f + learning_rate * layer_neuron (l, 0)->m_delta * layer_neuron (p, 0)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 0, 0), w));
    w = 0.21f + learning_rate * layer_neuron (l, 0)->m_delta * layer_neuron (p, 1)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 1, 0), w));
    w = 0.31f + learning_rate * layer_neuron (l, 0)->m_delta * layer_neuron (p, 2)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 2, 0), w));

    w = 0.12f + learning_rate * layer_neuron (l, 1)->m_delta * layer_neuron (p, 0)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 0, 1), w));
    w = 0.22f + learning_rate * layer_neuron (l, 1)->m_delta * layer_neuron (p, 1)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 1, 1), w));
    w = 0.32f + learning_rate * layer_neuron (l, 1)->m_delta * layer_neuron (p, 2)->m_output;
    g_assert_true (flt_equal (layer_neuron_weight_bp (p, 2, 1), w));

    /*
     * checking weights for input layer
     */

    dump_plane_bin (bin, mask_da);
    free_plane_bin (bin);
}

void test_1_hidden ()
{
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

    g_test_init (&argc, &argv, NULL);
    g_test_add_func ("/ann/test", test_all);
    return g_test_run ();
}
