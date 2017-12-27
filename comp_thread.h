#ifndef _COMP_THREAD_H_
#define _COMP_THREAD_H_

#include "ann.h"

/* A data closure structure to carry multiple variables between
 * threads. */
typedef struct _Msg2Consumer
{
    PlaneBin *m_pb;
    guint m_index;
} Msg2Consumer;

typedef struct _Msg2Producer
{
    PlaneBin *m_pb;
} Msg2Producer;


#define FUNC_ENTRY DBG ("in %s in thread %p ...\n", __func__, g_thread_self ())
#define FUNC_EXIT DBG ("ext %s ...\n", __func__)
#define DINT(x) DBG ("%s = %u\n", #x, (x))

void consumer_func (Msg2Consumer *data);

void consumer_func_data_free (Msg2Consumer *data);

void producer_func (Msg2Producer *data);

void producer_func_internal (PlaneBin *pb);

gboolean consumer_func_idle (gpointer user_data);

gboolean producer_func_idle (gpointer user_data);

void invoke_consumer_func (PlaneBin *pb, guint idx);

void init_threading (PlaneBin *pb, guint n_thread);

void start_threading (PlaneBin *pb);

void prepare_threading (PlaneBin *pb);

void invoke_producer_func (PlaneBin *pb);

void producer_func_data_free (Msg2Producer *data);

gpointer thread_main (gpointer user_data);

void producer_main (PlaneBin *pb);

#endif
