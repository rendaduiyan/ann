/*
 * threaded_func_test_mnist.c, apply threaded ANN to MNIST dataset
 * Copyright (c) 2018, haibolei <duiyanrenda@gmail.com>
 */

#include <stdlib.h>
#include <string.h>
#include "func_test_mnist.h"

void init_weights (PlaneBin *pb)
{
    init_weights_randomly (pb);
}

void free_sample_info (PlaneBin *pb)
{
    FUNC_ENTRY;
    SampleInfo *si = pb->m_loader;
    for (int i = 0; i < si->m_batch_size; i ++)
    {
        g_free (si->m_label_buf[i]);
        g_free (si->m_image_buf[i]);
        g_free (pb->m_in_data[i]);
        g_free (pb->m_exp_val[i]);
    }
    g_free (si->m_label_buf);
    g_free (si->m_image_buf);
    g_free (pb->m_in_data);
    g_free (pb->m_exp_val);
    g_free (si->m_image_buf_read);
    g_free (si->m_label_buf_read);
    fclose (si->m_image_fp);
    fclose (si->m_label_fp);
    g_free (si);
    pb->m_loader = NULL;
}

void read_header_info (PlaneBin *pb)
{
    SampleInfo *si = pb->m_loader;
    if (NULL != si->m_label_fp)
    {
        g_assert (1 == fread (&si->m_magic_num, 4, 1, si->m_label_fp));
        g_assert (1 == fread (&si->m_number_of_item, 4, 1, si->m_label_fp));
        NTOHL (si->m_magic_num);
        NTOHL (si->m_number_of_item);
    }
    if (NULL != si->m_image_fp)
    {
        g_assert (1 == fread (&si->m_magic_num, 4, 1, si->m_image_fp));
        g_assert (1 == fread (&si->m_number_of_item, 4, 1, si->m_image_fp));
        g_assert (1 == fread (&si->m_number_of_rows, 4, 1, si->m_image_fp));
        g_assert (1 == fread (&si->m_number_of_columns, 4, 1, si->m_image_fp));

        NTOHL (si->m_magic_num);
        NTOHL (si->m_number_of_item);
        NTOHL (si->m_number_of_rows);
        NTOHL (si->m_number_of_columns);
    }
}

gpointer init_train_mnist (PlaneBin *pb)
{
    FUNC_ENTRY;
    static gchar * const label_file = "train-labels-idx1-ubyte";
    static gchar * const image_file = "train-images-idx3-ubyte";
    SampleInfo *si = g_malloc (sizeof (SampleInfo));
    memset (si, 0, sizeof (SampleInfo));
    pb->m_loader = si;
    si->m_label_buf = g_malloc (sizeof (gdouble *) * pb->m_batch_size);
    si->m_image_buf = g_malloc (sizeof (gdouble *) * pb->m_batch_size);
    pb->m_in_data = g_malloc (sizeof (gdouble *) * pb->m_batch_size);
    pb->m_exp_val = g_malloc (sizeof (gdouble *) * pb->m_batch_size);
    si->m_batch_size = pb->m_batch_size;
    for (int i = 0; i < pb->m_batch_size; i ++)
    {
        si->m_label_buf[i] = g_malloc (sizeof (gdouble) * pb->m_curr_plane->m_on);
        pb->m_exp_val[i] = g_malloc (sizeof (gdouble) * pb->m_curr_plane->m_on);
        si->m_image_buf[i] = g_malloc (sizeof (gdouble) * pb->m_curr_plane->m_in);
        pb->m_in_data[i] = g_malloc (sizeof (gdouble) * pb->m_curr_plane->m_in);
    }
    if (NULL == si->m_image_fp)
    {
        si->m_image_fp = fopen (image_file, "rb");
        if (NULL == si->m_image_fp)
        {
            DBG ("failed to open file %s\n", image_file);
            return NULL;
        }
    }
    if (NULL == si->m_label_fp)
    {
        si->m_label_fp = fopen (label_file, "rb");
        if (NULL == si->m_label_fp)
        {
            DBG ("failed to open file %s\n", label_file);
            return NULL;
        }
    }
    read_header_info (pb);
    /*
     * ignore label header
     */
    //g_assert (si->m_magic_num == 2049);
    //g_assert (si->m_number_of_item == 60000);
    g_assert (si->m_magic_num == 2051);
    g_assert (si->m_number_of_item == 60000);
    g_assert (si->m_number_of_rows == 28);
    g_assert (si->m_number_of_columns == 28);
    pb->m_n_sample_all = si->m_number_of_item;
    si->m_label_buf_read = g_malloc (sizeof (guchar) * pb->m_batch_size);
    si->m_image_buf_read = g_malloc (sizeof (guchar) * pb->m_batch_size * 
            si->m_number_of_rows * si->m_number_of_columns);
    return si;
}

guint nb_train_mnist (PlaneBin *pb)
{
    FUNC_ENTRY;
    SampleInfo *si = pb->m_loader;
    si->m_num_in_buf = (si->m_n_sample_loaded + si->m_batch_size < si->m_number_of_item) 
        ? si->m_batch_size :  si->m_number_of_item - si->m_n_sample_loaded;
    g_assert (pb->m_curr_plane->m_on == 10);
    g_assert (pb->m_curr_plane->m_in == 28 * 28);
    DINT (si->m_num_in_buf);

        /*
         * reading labels
         */
    g_assert (si->m_num_in_buf == fread (si->m_label_buf_read, 1, 
                si->m_num_in_buf, si->m_label_fp));
    g_assert (si->m_num_in_buf == fread (si->m_image_buf_read, 
                si->m_number_of_rows * si->m_number_of_columns, 
                si->m_num_in_buf, si->m_image_fp));
        /*
         * reading images
         */
    guint index = 0;
    for (int i = 0; i < si->m_num_in_buf; 
            i ++, index += si->m_number_of_rows * si->m_number_of_columns)
    {
        for (guchar j = 0; j < (guchar)pb->m_curr_plane->m_on; j ++)
        {
            if (j != si->m_label_buf_read[i])
            {
                si->m_label_buf[i][j] = 0.0f;
            }
            else
            {
                si->m_label_buf[i][j] = 1.0f;
            }
        }
#ifdef DEBUG_DATA
        gchar buf[16];
        snprintf (buf, sizeof (buf)/sizeof (buf[0]), "%u.raw", i);
        FILE *dfp = fopen (buf, "w");
        fwrite (&si->m_image_buf_read[index], si->m_number_of_rows * si->m_number_of_columns, 1, dfp);
        fclose (dfp);
#endif
        for (int j = 0; j < pb->m_curr_plane->m_in; j ++)
        {
            si->m_image_buf[i][j] = (gdouble) si->m_image_buf_read[index + j] / 255.0f; 
        }
    }
    return si->m_num_in_buf;
}

void copy_sample (PlaneBin *pb)
{
    FUNC_ENTRY;
    SampleInfo *si = pb->m_loader;
    pb->m_n_sample = si->m_num_in_buf;
    for (int i = 0; i < si->m_num_in_buf; i ++)
    {
        memcpy (pb->m_in_data[i], si->m_image_buf[i], pb->m_curr_plane->m_in * sizeof (gdouble));
        memcpy (pb->m_exp_val[i], si->m_label_buf[i], pb->m_curr_plane->m_on * sizeof (gdouble));
    }
}

gpointer init_test_mnist (PlaneBin *pb)
{
    FUNC_ENTRY;
    SampleInfo *si = pb->m_loader;
    /*
     * there is no different in format between training data set and test data set
     * since the size of test data set is less than that of training data set
     * it's okay to reuse the structures here
     */
    static gchar * const label_file = "t10k-labels-idx1-ubyte";
    static gchar * const image_file = "t10k-images-idx3-ubyte";
    /*
     * file not closed yet
     */
    if (NULL != si->m_image_fp)
    {
        fclose (si->m_image_fp);
    }
    si->m_image_fp = fopen (image_file, "rb");
    if (NULL == si->m_image_fp)
    {
        DBG ("failed to open file %s\n", image_file);
        return NULL;
    }
    if (NULL != si->m_label_fp)
    {
        fclose (si->m_label_fp);
    }
    si->m_label_fp = fopen (label_file, "rb");
    if (NULL == si->m_label_fp)
    {
        DBG ("failed to open file %s\n", label_file);
        return NULL;
    }
    read_header_info (pb);
    g_assert (si->m_magic_num == 2051);
    g_assert (si->m_number_of_item == 10000);
    g_assert (si->m_number_of_rows == 28);
    g_assert (si->m_number_of_columns == 28);
    pb->m_n_sample_all = si->m_number_of_item;
    return si;
}

void test_mnist (PlaneBin *pb)
{
    /*
     * checkpoint here for training
     */
    gint64 end_time = g_get_real_time () / 1000;
    g_print ("it cost %03lld (ms) to train %u sample(s) %u time(s) with %u neuron(s) in hidden layer\n", (end_time - pb->m_start_time), pb->m_n_sample_all, pb->m_n_iteration, pb->m_curr_plane->m_hn);

    SampleInfo *si = init_test_mnist (pb);
    guint n_correct_answer = 0;
    guint n_iter = si->m_number_of_item / si->m_batch_size;
    for (guint i = 0; i < n_iter; i ++)
    {
        nb_train_mnist (pb);
        for (guint j = 0; j < si->m_num_in_buf; j ++)
        {
            load_data (pb, si->m_image_buf[j]);
            fwd_plane_bin_act (pb);
            gdouble max_o = 0.0f;
            Layer *ol = pb->m_curr_plane->m_layers[pb->m_curr_plane->m_ln - 1];
            guint target = 0;
            guint expect = 0;
            for (guint k = 0; k < pb->m_curr_plane->m_on; k ++)
            {
#ifdef DEBUG_DATA
                DBG ("%g ", layer_neuron (ol, k)->m_output);
#endif
                if (layer_neuron (ol, k)->m_output > max_o)
                {
                    max_o = layer_neuron (ol, k)->m_output;
                    target = k;
                }
            }
            for (guint k = 0; k < pb->m_curr_plane->m_on; k ++)
            {
                if (flt_equal (si->m_label_buf[j][k], 1.0f))
                {
                    expect = k;
                    break;
                }
            }
            if (expect == target)
            {
                n_correct_answer ++;
            }
        }
    }
#ifdef DEBUG_DATA
    NL;
#endif
    g_print ("correct ratio: %g\n", (gdouble ) n_correct_answer * 100 / si->m_number_of_item);
}

int main (int argc, char **argv)
{
    guint plane_thread_num = 4;
    if (argc == 2)
    {
        plane_thread_num = atoi (argv[1]);
    }


    static const gdouble learning_rate = 0.8f;
    static const guint training_iter = 1;
    static const guint in = 28 *28;
    static const guint hn = 10 * 10;
    static const guint on = 10;
    static const guint hln = 1;
    static const guint batch = 100;

    PlaneBin *bin = construct_plane_bin (1, in, hn, on, hln);
    bin->m_cbs.m_check_fwd_cb = NULL;
    bin->m_cbs.m_check_bwd_cb = NULL;
    bin->m_cbs.m_test_cb = (test_after_training) test_mnist;
    bin->m_cbs.m_next_batch_cb = (next_batch) nb_train_mnist;
    bin->m_cbs.m_init_loading_cb = (init_loading) init_train_mnist;
    bin->m_cbs.m_copy_data_cb = (copy_data) copy_sample;
    bin->m_cbs.m_free_cb = (free_data) free_sample_info;

    //set_plane_func (bin->m_plane[0], act_relu, dact_relu);
    //set_plane_func (bin->m_plane[0], act_threshold, dact_threshold);


    bin->m_n_iteration = training_iter;
    bin->m_curr_iter = 0;
    bin->m_lr = learning_rate;
    bin->m_batch_size = batch;

    init_weights (bin);

    init_threading (bin, plane_thread_num);
    bin->m_start_time = g_get_real_time () / 1000;
    /*
     * trigger training
     */
    start_threading (bin);
    producer_main (bin);
    free_plane_bin (bin);

    return 0;
}
