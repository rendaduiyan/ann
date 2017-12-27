#include <string.h>
#include "comp_thread.h"

gpointer thread_main (gpointer user_data)
{
    FUNC_ENTRY;
    GMainContext *consumer = user_data;
    GMainLoop *main_loop;

    /* Set up the thread’s context and run it forever. */
    g_main_context_push_thread_default (consumer);

    main_loop = g_main_loop_new (consumer, FALSE);
    g_main_loop_run (main_loop);
    g_main_loop_unref (main_loop);

    g_main_context_pop_thread_default (consumer);
    g_main_context_unref (consumer);

    FUNC_EXIT;
    return NULL;
}

void producer_func_data_free (Msg2Producer *data)
{
    FUNC_ENTRY;
    g_free (data);
}

void consumer_func_data_free (Msg2Consumer *data)
{
    FUNC_ENTRY;
    /*
     * the neurons in the queue can be freed in main thread
     * nothing needs to be done here
     */
    g_free (data);
}

void consumer_func (Msg2Consumer *data)
{
    FUNC_ENTRY;
    g_assert (g_main_context_is_owner (data->m_pb->m_contexts [data->m_index]));
    gboolean running = TRUE;
    Neuron *n = NULL;
    while ((n = g_async_queue_try_pop (data->m_pb->m_q_units)) != NULL)
    {
        switch (data->m_pb->m_df)
        {
            case df_fwd: fwd_neuron_act (n); break;
            case df_bwd: 
                         {
                             bp_neuron_act (n); 
                             bp_neuron_act_weight (n, data->m_pb->m_lr); 
                             break;
                         }
        }
    }
    if (g_atomic_int_dec_and_test (&data->m_pb->m_n_done))
    {
        invoke_producer_func (data->m_pb);
    }
    FUNC_EXIT;
}

/* Convert an idle callback into a call to consumer_func(). */
gboolean consumer_func_idle (gpointer user_data)
{
    FUNC_ENTRY;

    consumer_func (user_data);

    return G_SOURCE_REMOVE;
}

void producer_func (Msg2Producer *data)
{
    producer_func_internal (data->m_pb);
}

void producer_func_internal (PlaneBin *pb)
{
    FUNC_ENTRY;
    /*
     * computation is done
     */
    if (pb->m_df == df_fwd)
    {
        /*
         * output layer is done
         * starting the bp routine
         * since all threads have called this callback
         * none of them will try to change the variables
         */
        if (pb->m_check_func[0] != NULL)
        {
            pb->m_check_func[0] (pb);
        }
        /*
         * change to bp and then compute for output layer
         */
        bp_layer_act_exp (pb->m_curr_plane->m_layers[pb->m_w_num], 
                pb->m_exp_val[pb->m_curr_sample]);
        pb->m_df = df_bwd;
        start_threading (pb);
    }
    else
    {
        /*
         * bp is done for a sample
         */
        if (pb->m_check_func[1] != NULL)
        {
            pb->m_check_func[1] (pb);
        }
        pb->m_curr_sample ++;
        if (pb->m_curr_sample < pb->m_n_sample)
        {
            for (int i = 0; i < 2; i ++)
            {
                DBG ("%g ", pb->m_in_data[pb->m_curr_sample][i]);
            }
            NL;
            load_data (pb, pb->m_in_data[pb->m_curr_sample]);
            pb->m_df = df_fwd;
            start_threading (pb);
        }
        else
        {
            /*
             * traing for this iteration is done
             */
            if (pb->m_curr_iter < pb->m_n_iteration)
            {
                pb->m_curr_iter ++;
                pb->m_df = df_fwd;
                /*
                 * always start from hidden layer
                 */
                pb->m_curr_sample = 0;
                load_data (pb, pb->m_in_data[pb->m_curr_sample]);
                start_threading (pb);
            }
            else
            {
                /*
                 * traing is done
                 * test can be triggered now
                 */
                if (pb->m_test != NULL)
                {
                    pb->m_test (pb);
                }
                /*
                 * clean up
                 */
                DBG ("to clean up %u...\n", pb->m_n_thread);
                g_main_loop_quit (pb->m_loop);
                for (int i = 0; i < pb->m_n_thread; i ++)
                {
                    g_thread_unref (pb->m_workers[i]);
                }
                g_free (pb->m_workers);
                g_free (pb->m_holder->m_inputs[0]);
                g_free (pb->m_holder->m_weights[0]);
                g_free (pb->m_holder->m_inputs_bp[0]);
                g_free (pb->m_holder->m_weights_bp[0]);
                free_neuron (pb->m_holder);
                g_free (pb->m_contexts);
                g_free (pb->m_q_units);
            }
        }
    }
}

gboolean producer_func_idle (gpointer user_data)
{
    FUNC_ENTRY;
    producer_func (user_data);
    return G_SOURCE_REMOVE;
}

void invoke_producer_func (PlaneBin *pb)
{
    FUNC_ENTRY;
    GSource *idle_source;
    Msg2Producer *data;

    data = g_new0 (Msg2Producer, 1);
    data->m_pb = pb;
    idle_source = g_idle_source_new ();
    g_source_set_callback (idle_source, producer_func_idle, data,
            (GDestroyNotify) producer_func_data_free);
    g_source_set_priority (idle_source, G_PRIORITY_DEFAULT);
    g_source_attach (idle_source, pb->m_contexts[pb->m_n_thread]);
    g_source_unref (idle_source);
}

/* Function to be called in the main thread to schedule a call to
 * consumer_func() in thread, passing the given parameters along. */
void invoke_consumer_func (PlaneBin *pb, guint idx)
{
    FUNC_ENTRY;
    GSource *idle_source;
    Msg2Consumer *data;

    /* Create a data closure to pass all the desired variables
     * between threads. */
    data = g_new0 (Msg2Consumer, 1);
    data->m_index = idx;
    data->m_pb = pb;

    /* Create a new idle source, set consumer_func() as the callback with
     * some data to be passed between threads, bump up the priority
     * and schedule it by attaching it to thread’s context. */
    idle_source = g_idle_source_new ();
    g_source_set_callback (idle_source, consumer_func_idle, data,
            (GDestroyNotify) consumer_func_data_free);
    g_source_set_priority (idle_source, G_PRIORITY_DEFAULT);
    g_source_attach (idle_source, pb->m_contexts[idx]);
    g_source_unref (idle_source);
}

/* function for the main thread to assign the tasks. */
void init_threading (PlaneBin *pb, guint n_thread)
{
    FUNC_ENTRY;
    GThread *thread;
    gchar thread_name[16];

    /*
     * each thread has its own context
     * the last one is for producer thread
     * which means n_thread + 1
     */

    pb->m_holder = construct_neuron (1, 0);
    pb->m_holder->m_inputs[0] = g_malloc (sizeof (gdouble));
    *pb->m_holder->m_inputs[0] = 1.0f;
    pb->m_holder->m_weights[0] = g_malloc (sizeof (gdouble));
    *pb->m_holder->m_weights[0] = 1.0f;
    pb->m_holder->m_size_bp = 1;
    pb->m_holder->m_inputs_bp = g_malloc (sizeof (gdouble*) * pb->m_holder->m_size_bp);
    pb->m_holder->m_inputs_bp[0] = g_malloc (sizeof (gdouble));
    *pb->m_holder->m_inputs_bp[0] = 1.0f;
    pb->m_holder->m_weights_bp = g_malloc (sizeof (gdouble*) * pb->m_holder->m_size_bp);
    pb->m_holder->m_weights_bp[0] = g_malloc (sizeof (gdouble));
    *pb->m_holder->m_weights_bp[0] = 1.0f;

    pb->m_n_thread = n_thread;
    pb->m_n_done = n_thread;
    pb->m_contexts = g_malloc (sizeof (GMainContext *) * (n_thread + 1));
    pb->m_workers = g_malloc (sizeof (GThread *) * n_thread);
    pb->m_q_units = g_async_queue_new ();
    pb->m_curr_sample = 0;
    pb->m_df = df_fwd;
    load_data (pb, pb->m_in_data[pb->m_curr_sample]);


    for (int i = 0; i < n_thread; i ++)
    {
        /* Spawn a background thread and pass it a reference to its
         * GMainContext. Retain a reference for use in this thread
         * too. */
        guint name_len = sizeof (thread_name)/sizeof (thread_name[0]);
        memset (thread_name, 0, name_len);
        g_snprintf (thread_name, name_len, "thread_%u", i);
        pb->m_contexts[i] = g_main_context_new ();
        pb->m_workers[i] = g_thread_new (thread_name, thread_main,
                g_main_context_ref (pb->m_contexts[i]));
    }
    /*
     * context needs to be passed to consumer threads 
     * so that they can invoke the function in producer thread
     */
    pb->m_loop = g_main_loop_new (NULL, FALSE);
    pb->m_contexts[n_thread] = g_main_context_get_thread_default ();
}

void start_threading (PlaneBin *pb)
{
    FUNC_ENTRY;
    prepare_threading (pb);
    guint n_units = g_async_queue_length (pb->m_q_units);
    DBG ("neurons in the queue: %d\n", n_units);
    guint n_thread = pb->m_n_thread > n_units ? n_units : pb->m_n_thread;
    pb->m_n_done = n_thread;
    for (guint i = 0; i < n_thread; i ++)
    {
        invoke_consumer_func (pb, i);
    }
}

void prepare_threading (PlaneBin *pb)
{
    FUNC_ENTRY;
    /*
     * for now, there is only one plane
     */
    g_assert (pb->m_in_data != NULL);
    g_assert (pb->m_n_sample > 0);
    DBG ("%s\n", (pb->m_df == df_fwd) ? "->" : "<-");
    DINT (pb->m_curr_iter);
    DINT (pb->m_curr_sample);
    switch (pb->m_df)
    {
        case df_fwd:
            {
                /*
                 * starting from hidden layer
                 */
                for (int i = 1; i < pb->m_curr_plane->m_ln; i ++)
                {
                    Layer *l = pb->m_curr_plane->m_layers[i];
                    for (int j = 0; j < layer_raw_size (l); j ++)
                    {
                        g_async_queue_push (pb->m_q_units, layer_neuron (l, j));
                    }
                    for (int j = 0; j < (pb->m_n_thread - layer_raw_size (l) % pb->m_n_thread); j ++)
                    {
                        g_async_queue_push (pb->m_q_units, pb->m_holder);
                    }
                    for (int j = 0; j < pb->m_n_thread; j ++)
                    {
                        g_async_queue_push (pb->m_q_units, pb->m_holder);
                    }
                }
                break;
            }
        case df_bwd:
            {
                for (int i = pb->m_curr_plane->m_ln - 1 - 1; i >= 0; i --)
                {
                    Layer *l = pb->m_curr_plane->m_layers[i];
                    for (int j = 0; j < layer_size (l); j ++)
                    {
                        g_async_queue_push (pb->m_q_units, layer_neuron (l, j));
                    }
                    for (int j = 0; j < (pb->m_n_thread - layer_size (l) % pb->m_n_thread); j ++)
                    {
                        g_async_queue_push (pb->m_q_units, pb->m_holder);
                    }
                    for (int j = 0; j < pb->m_n_thread; j ++)
                    {
                        g_async_queue_push (pb->m_q_units, pb->m_holder);
                    }
                }
                break;
            }
    }

}

void producer_main (PlaneBin *pb)
{
    if (!g_main_loop_is_running (pb->m_loop))
    {
        g_main_loop_run (pb->m_loop);
    }
    DBG ("main loop quited ...\n");
    g_main_loop_unref (pb->m_loop);
}


