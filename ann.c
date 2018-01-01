#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ann.h"

gchar * const sep = "--------------------------------";
gchar * const ln_input = "input";
gchar * const ln_hidden = "hidden";
gchar * const ln_output = "output";


guint layer_size (Layer *l)
{
    return l->m_size;
}

Neuron *layer_neuron (Layer *l, guint i)
{
    return l->m_neurons[i];
}

gdouble **layer_neuron_w_bp (Layer *l, guint i, guint j)
{
    return &(l->m_neurons[i]->m_weights_bp[j]);
}

gdouble layer_neuron_weight (Layer *l, guint i, guint j)
{
    return *(l->m_neurons[i]->m_weights[j]);
}

gdouble layer_neuron_weight_bp (Layer *l, guint i, guint j)
{
    return *(l->m_neurons[i]->m_weights_bp[j]);
}

gdouble **layer_neuron_w (Layer *l, guint i, guint j)
{
    return &(l->m_neurons[i]->m_weights[j]);
}

gsize layer_input_scale (Layer *l)
{
    return l->m_scale;
}

gdouble **neuron_input (Neuron *n, guint i)
{
    return &(n->m_inputs[i]);
}

void free_neuron (Neuron *n)
{
    if (n->m_size > 0)
    {
        g_free (n->m_inputs);
        g_free (n->m_weights);
    }
    if (n->m_size_bp > 0)
    {
        g_free (n->m_inputs_bp);
        g_free (n->m_weights_bp);
    }
    g_free (n);
}

void free_layer (Layer *l)
{
    for (int i = 0; i < l->m_size; i ++)
    {
        free_neuron (l->m_neurons[i]);
    }
    g_free (l->m_neurons);
    g_free (l);
}

void free_plane (Plane *plane)
{
    g_free (plane->m_weights);

    for (int i = 0; i < plane->m_ln; i ++)
    {
        free_layer (plane->m_layers[i]);
    }
    g_free (plane->m_layers);
    g_free (plane);
}

void set_plane_func (Plane *plane, actviation_func func, delta_actviation_func dfunc)
{
    for (int i = 0; i < plane->m_ln; i ++)
    {
        set_layer_func (plane->m_layers[i], func, dfunc);
    }
}

void set_layer_func (Layer* l, actviation_func func, delta_actviation_func dfunc)
{
    for (int i = 0; i < layer_size (l); i ++)
    {
        layer_neuron (l, i)->m_act_func = func;
        layer_neuron (l, i)->m_dact_func = dfunc;
    }
}

/*
 * deritive sigmoid: takes a real-valued input and squashes it to range between 0 and 1
 */
gdouble dact_sigmoid (gdouble x)
{
    return (1.0f - x) * x;
}
/*
 * sigmoid: takes a real-valued input and squashes it to range between 0 and 1
 */
gdouble act_sigmoid (gdouble x)
{
    /*
     * no need to calc for all the range
     * approximately picking 45.0 is a threadshold
    if (x < -45.0f) 
    {
        return 0;
    }
    if (x > 45.0f) 
    {
        return 1;
    }
    */
    return 1.0f / (1.0f + exp (-x));
}

gdouble act_threshold (gdouble x)
{
    return (x > 0 ? 1.0f : 0.0f);
}

gdouble dact_threshold (gdouble x)
{
    return (x > 0 ? 1.0f : 0.0f);
}

gdouble dact_relu (gdouble x)
{
    x = x;
    if (x > 0)
    {
        return 1;
    }
    return 0;
}

/*
 * relu: stands for Rectified Linear Unit. It takes a real-valued iput and thresholds it at zero
 */
gdouble act_relu (gdouble x)
{
    if (x > 0)
    {
        return x;
    }
    return 0;
}

/*
 * @n is the number of features X, X1, X2, ... Xn
 * @id is the identifier for this neuron
 * @return the new Neuron created
 */

Neuron *construct_neuron (guint n, guint id)
{
    Neuron *node = g_malloc (sizeof (Neuron));
    node->m_size = n;
    node->m_id = id;
    node->m_act_func = act_sigmoid;
    if (n > 0)
    {
        node->m_inputs = g_malloc (sizeof (gdouble *) * n);
        node->m_weights = g_malloc (sizeof (gdouble *) * n);
        for (int i = 0; i < n; i ++)
        {
            node->m_inputs[i] = NULL;
            node->m_weights[i] = NULL;
        }
        node->m_output = 0.0f;
    }
    else
    {
        node->m_output = 1.0f;
    }
    node->m_delta = 0.0f;
    node->m_size_bp = 0;
    node->m_inputs_bp = NULL;
    node->m_weights_bp = NULL;
    node->m_dact_func = dact_sigmoid;
    return node;
}

/*
 * @n is the number of nodes in this layer
 * excluding the bias node
 * @m is the nubmer of features, X1, X2, ... Xn
 * @id is the identifier for this layer
 * @with_bias, if there is a bias neuron
 * @return the new layer created
 */

Layer *construct_layer (guint n, guint m, guint id, gboolean with_bias, layer_type lt)
{
    int i = 0;
    Layer *layer = g_malloc (sizeof (Layer));
    layer->m_lt = lt;
    layer->m_id = id;
    layer->m_scale = m;
    layer->m_with_bias = with_bias;
    layer->m_size = with_bias ? n + 1 : n;
    layer->m_neurons = g_malloc (sizeof (Neuron*) *layer->m_size);
    while (i < n)
    {
        layer->m_neurons[i] = construct_neuron (m, i);
        layer->m_neurons[i]->m_layer_id = layer->m_id;
        i ++;
    }
    if (with_bias)
    {
        layer->m_neurons[i] = construct_neuron (0, 0);
        layer->m_neurons[i]->m_layer_id = layer->m_id;
    }
    return layer;
}


/*
 * @in is the number of nodes in input layer
 * @hn is the number of nodes in hidden layer
 * @on is the number of nodes in output layer
 * @hln is the number of hidden layers; hln >= 1
 * @id is identifier for this plane
 * @return, new plane instance
 * Bias is a special neuron without input (fixed to 1) at the tail of the neuron list
 * TO-DO: decdie wether or not to support 0 hidden layer
 */

Plane *construct_plane (guint in, guint hn, guint on, guint hln, guint id)
{
    Plane *plane = g_malloc (sizeof (Plane));
    plane->m_in = in;
    plane->m_on = on;
    plane->m_hn = hn;
    plane->m_id = id;
    plane->m_ln = hln + 2; //input layer + output layer
    plane->m_layers = g_malloc (sizeof (Layer *) * plane->m_ln);
    guint layer_id = 0;
    plane->m_layers[layer_id] = construct_layer (in, 1, layer_id, TRUE, lt_input);
    gsize prev_scale = layer_size (plane->m_layers[layer_id]);
    layer_id ++;
    for (int i = 0; i < hln; i ++, layer_id ++)
    {
        plane->m_layers[layer_id] = construct_layer (hn, prev_scale, layer_id, TRUE, lt_hidden);
        prev_scale = layer_size (plane->m_layers[layer_id]);
    }
    plane->m_layers[layer_id] = construct_layer (on, prev_scale, layer_id, FALSE, lt_output);

    plane->m_weights = g_malloc (sizeof (gdouble **) * (plane->m_ln - 1));

    link_plane_layers (plane);

    return plane;
}

void init_plane_bp (Plane *fwd)
{
    /*
     * output layer
     */
    Layer *p = fwd->m_layers[fwd->m_ln - 1];
    /*
     * starting from last hidden layer
     */
    for (int i = fwd->m_ln - 1 - 1; i >= 0; i --)
    {
        Layer *l = fwd->m_layers[i];
        for (int j = 0; j < layer_size (l); j ++)
        {
            Neuron *n = layer_neuron (l, j);
            n->m_size_bp = layer_raw_size (p);
            n->m_inputs_bp = g_malloc (n->m_size_bp * sizeof (gdouble*));
            n->m_weights_bp = g_malloc (n->m_size_bp * sizeof (gdouble*));
            for (int k = 0; k < n->m_size_bp; k ++)
            {
                n->m_inputs_bp[k] = &layer_neuron (p, k)->m_delta;
            }
        }
        assign_weights_to_layer_bp (fwd->m_layers[i], &(fwd->m_dims[i]), fwd->m_weights[i]);
        p = l;
    }
}

void init_weights_randomly (PlaneBin *pb)
{
    GRand *grand = g_rand_new ();
    for (int i = 0; i < pb->m_w_num; i ++)
    {
        for (int j = 0; j < pb->m_dims[i].m_x; j ++)
        {
            for (int k = 0; k < pb->m_dims[i].m_y; k ++)
            {
                pb->m_weights[i][j][k] = g_rand_double (grand) - 0.5f;
            }
        }
    }
    g_rand_free (grand);
}


gboolean link_layers (Layer *l1, Layer *l2)
{
    DBG ("linking layers %u and %u ...\n", l1->m_id, l2->m_id);
    /*
     * the layers in the middle will be dumped twicly
    */

    gboolean ret_val = TRUE;
    if (layer_size (l1) != layer_input_scale (l2))
    {
        DBG ("number of l1's output is NOT aligned to l2's input");
        ret_val = FALSE;
    }
    /*
     * for second layer, bias neuron is always not linked
     */
    for (int j = 0; j < layer_raw_size (l2); j ++)
    {
        for (int i = 0; i < layer_size (l1); i ++)
        {
            *neuron_input (layer_neuron (l2, j), i) = &(layer_neuron (l1, i)->m_output);
        }
    }
    return ret_val;
}

gboolean link_plane_layers (Plane *plane)
{
    gboolean ret_val = TRUE;
    Layer *prev = plane->m_layers[0];
    for (int i = 1; i < plane->m_ln; i ++)
    {
        ret_val = link_layers (prev, plane->m_layers[i]);
        if (!ret_val)
        {
            break;
        }
        prev = plane->m_layers[i];
    }
    return ret_val;
}

void free_array (gdouble **arr, guint s)
{
    for (int i = 0; i < s; i ++)
    {
        g_free (arr[i]);
    }
    g_free (arr);
}

void free_plane_bin (PlaneBin *pb)
{
    for (int i = 0; i < pb->m_w_num; i ++)
    {
        free_array (pb->m_weights[i], pb->m_dims[i].m_x);
    }
    g_free (pb->m_weights);
    g_free (pb->m_dims);

    for (int i = 0; i < pb->m_size; i ++)
    {
        free_plane (pb->m_plane[i]);
    }
    g_free (pb->m_plane);
    g_free (pb);
}

void dump_weights (gdouble ***w, ArraySize *as, guint ln)
{
    DBG ("weights: \n");
    for (int i = 0; i < ln; i ++)
    {
        DBG ("{%d, %d, %d}\n", i, as[i].m_x, as[i].m_y);
        for (int j = 0; j < as[i].m_x; j ++)
        {
            for (int k = 0; k < as[i].m_y; k ++)
            {
                DBG ("%g ", w[i][j][k]);
            }
            NL;
        }
    }
}

void dump_plane_bin (PlaneBin *pb, guint dump_bits)
{
    DBG ("PlaneBin %p Info:\n", pb);
    SEP;
    for (int i = 0; i < pb->m_size; i ++)
    {
        if (dump_bits & mask_db)
        {
            DBG ("planes %u:\n", i);
        }
        dump_plane (pb->m_plane[i], dump_bits);
    }
    if (dump_bits & mask_dw)
    {
        DBG ("PlaneBin's weights: \n");
        dump_weights (pb->m_weights, pb->m_dims, pb->m_w_num);
        SEP;
    }
}

void dump_plane (Plane *plane, guint dump_bits)
{
    SEP;
    for (int i = 0; i < plane->m_ln; i ++)
    {
        if (dump_bits & mask_db)
        {
            DBG ("hidden layer %u, %p:\n", i, plane->m_layers[i]);
        }
        dump_layer (plane->m_layers[i], dump_bits);
    }
    SEP;
}

gchar *layer_name (layer_type lt)
{
    gchar *ret_val = NULL;
    switch (lt)
    {
        case lt_input: ret_val = ln_input; break;
        case lt_hidden: ret_val = ln_hidden; break;
        case lt_output: ret_val = ln_output; break;
    };
    return ret_val;
}

void dump_layer (Layer *l, guint dump_bits)
{
    if (dump_bits & mask_db)
    {
        DBG ("layer %s %u, neuron: %u, scale of feature: %u\n", layer_name (l->m_lt), l->m_id, l->m_size, l->m_scale);
    }
    SEP;
    for (int i = 0; i < l->m_size; i ++)
    {
        if (dump_bits & mask_db)
        {
            DBG ("neuron %u, %p\n", i, l->m_neurons[i]);
        }
        if (l->m_id > 0)
        {
            dump_neuron (l->m_neurons[i], dump_bits);
        }
    }
    SEP;
}

void dump_neuron (Neuron *n, guint dump_bits)
{
    if (dump_bits & mask_db)
    {
        DBG ("id:%u, scale of feature X:%u\n", n->m_id, n->m_size);
    }
    SEP;
    if (dump_bits & mask_dw)
    {
        DBG ("neuron weights:\n");
        SEP;
        for (int i = 0; i < n->m_size; i ++)
        {
            if (n->m_weights[i])
            {
                DBG ("%g ", *(n->m_weights[i]));
            }
        }
        NL;
        SEP;
    }
    if (dump_bits & mask_di)
    {
        DBG ("neuron inputs:\n");
        SEP;
        for (int i = 0; i < n->m_size; i ++)
        {
            DBG ("%g ", *n->m_inputs[i]);
        }
        NL;
        SEP;
    }
    if (dump_bits & mask_do)
    {
        DBG ("neuron outputs: %g\n", n->m_output);
    }
    if (dump_bits & mask_dd)
    {
        DBG ("neuron delta: %g\n", n->m_delta);
    }
    SEP;
}

/*
 * @pn, planes in the bin
 * @in, neurons in input layer
 * @hn, neurons in all hidden layer
 * @on, neurons in output layer
 * @hl, hidden layers
 * planes in a bin will share the weights
 */
PlaneBin *construct_plane_bin (guint pn, guint in, guint hn, guint on, guint hl)
{
    PlaneBin *pb = g_malloc (sizeof (PlaneBin));
    pb->m_size = pn;
    pb->m_plane = g_malloc (sizeof (Plane *) * pn);
    for (int i = 0; i < pn; i ++)
    {
        pb->m_plane[i] = construct_plane (in, hn, on, hl, i);
    }

    /*
     * Constructing the weights planes
     * the total number of weights vector is 1 (input) + (hidden layers - 1) + output
     */
    pb->m_w_num = hl - 1 + 2;
    pb->m_dims = g_malloc (sizeof (ArraySize) * pb->m_w_num);
    pb->m_weights = g_malloc (sizeof (gdouble **) * pb->m_w_num); 

    /* 
     * scale of weight vector between input and first hidden layer:
     * (in + 1) x hn 
     *
     */
    pb->m_dims [0].m_x = in + 1;
    pb->m_dims [0].m_y = hn;
    pb->m_weights[0] = g_malloc (sizeof (gdouble*) * pb->m_dims[0].m_x);
    for (int i = 0; i < pb->m_dims [0].m_x; i ++)
    {
        pb->m_weights[0][i] = g_malloc (sizeof (gdouble) * pb->m_dims [0].m_y);
    }

    /*
     * Constructing the weights between hidden layers:
     * (hn + 1) x (hn)
     */
    for (int i = 0; i < hl - 1; i ++)
    {
        pb->m_dims[i + 1].m_x = hn + 1;
        pb->m_dims[i + 1].m_y = hn;
        pb->m_weights[i + 1] = g_malloc (sizeof (gdouble*) * pb->m_dims[i + 1].m_x);
        for (int j = 0; j < pb->m_dims[i + 1].m_x; j ++)
        {
            pb->m_weights[i + 1][j] = g_malloc (sizeof (gdouble) * pb->m_dims[i + 1].m_y);
        }
    }

    /*
     * Constructin the weights between last hidden layer and output layer:
     * (hn + 1) x on
     */
    pb->m_dims[hl].m_x = hn + 1;
    pb->m_dims[hl].m_y = on;
    pb->m_weights[hl] = g_malloc (sizeof (gdouble *) * pb->m_dims[hl].m_x);
    for (int i = 0; i < pb->m_dims[hl].m_x; i ++)
    {
        pb->m_weights[hl][i] = g_malloc (sizeof (gdouble) * pb->m_dims[hl].m_y);
    }

    for (int i = 0; i < pn; i ++)
    {
        pb->m_plane[i]->m_dims = pb->m_dims;
        for (int j = 0; j < pb->m_w_num; j ++)
        {
            pb->m_plane[i]->m_weights[j] = pb->m_weights[j];
        }
        assign_weights_to_neuron (pb->m_plane[i]);
        init_plane_bp (pb->m_plane[i]);
    }

    memset (&pb->m_cbs, 0, sizeof (Callbacks));
    pb->m_curr_plane = pb->m_plane[0];
    pb->m_loader = NULL;

    return pb;
}

gsize layer_raw_size (Layer *l)
{
    return layer_size (l) - ((l->m_with_bias) ? 1 : 0);
}


void assign_weights_to_layer_bp (Layer *l, ArraySize *as, gdouble **w)
{
    int n_neurons = layer_size (l);
    DBG ("iterating %u neuron(s) - (%u, %u) ...\n", n_neurons, as->m_x, as->m_y);
    for (int j = 0; j < n_neurons; j ++)
    {
        for (int k = 0; k < as->m_y; k ++)
        {
            *layer_neuron_w_bp (l, j, k) = &(w[j][k]);
        }
    }
}

void assign_weights_to_layer (Layer *l, ArraySize *as, gdouble **w)
{
    int n_neurons = layer_raw_size (l);
    DBG ("iterating %u neurons ...\n", n_neurons);
    for (int j = 0; j < n_neurons; j ++)
    {
        DBG ("iterating {%u, %u} input ...\n", as->m_x, as->m_y);
        for (int k = 0; k < as->m_x; k ++)
        {
            *layer_neuron_w (l, j, k) = &(w[k][j]);
        }
    }
}

void assign_weights_to_neuron (Plane *plane)
{
    /*
     * starting from the first hidden layer
     * ending at the last one
     */
    for (int i = 1; i < plane->m_ln; i ++)
    {
        assign_weights_to_layer (plane->m_layers[i], &(plane->m_dims[i - 1]), plane->m_weights[i - 1]);
    }
}

void fwd_neuron_act (Neuron *n)
{
    gdouble sum = 0.0f;
    for (int i = 0; i < n->m_size; i ++)
    {
        sum += ((*n->m_inputs[i]) * (*n->m_weights[i]));
    }
    n->m_output = n->m_act_func (sum);
}

void fwd_layer_act (Layer *l)
{
    for (int i = 0; i < layer_raw_size (l); i ++)
    {
        fwd_neuron_act (layer_neuron (l, i));
    }
}


void bp_layer_weight (Layer *l1, Layer *l2, gdouble lr)
{
    for (int i = 0; i < layer_size (l1); i ++)
    {
        Neuron *n1 = layer_neuron (l1, i);
        for (int j = 0; j < layer_raw_size (l2); j ++)
        {
            Neuron *n2 = layer_neuron (l2, j);
            gdouble *w = *layer_neuron_w (l2, j, i);
            *w += (lr * n2->m_delta * n1->m_output);

        }
    }
}

void bp_layer_act_bp (Layer *l1, Layer *l2)
{
    for (int i = 0; i < layer_raw_size(l1); i ++)
    {
        Neuron *n1 = layer_neuron (l1, i);
        gdouble *o = &n1->m_output;
        gdouble delta = 0.0f;
        for (int j = 0; j < layer_raw_size (l2); j ++)
        {
            Neuron *n2 = layer_neuron (l2, j);
            delta += n2->m_delta * *n2->m_weights_bp[i];
        }
        n1->m_delta = n1->m_dact_func (*o) * delta;
    }
}

void bp_layer_act_exp (Layer *l, gdouble *exp)
{
    for (int i = 0; i < layer_size (l); i ++)
    {
        Neuron *n = layer_neuron (l, i);
        gdouble o = n->m_output;
        //g_print ("%g ", exp[i]);
#ifdef DEBUG_DATA
        DBG ("%g ", o);
#endif
        n->m_delta = (exp[i] - o) * n->m_dact_func (o);
    }
    //g_print ("\n");
#ifdef DEBUG_DATA
    NL;
    DBG ("expectation:");
    for (int i = 0; i < layer_size (l); i ++)
    {
        DBG ("%g ", exp[i]);
    }
    NL;
#endif
}

void bp_plane_act (Plane *plane, gdouble *exp, gdouble lr)
{
    /*
     * Getting output layer deltas
     */
    bp_layer_act_exp (plane->m_layers[plane->m_ln - 1], exp);

    /*
     * back properation to hidden layers
     */
    for (int i = plane->m_ln - 1 - 1; i >= 0; i --)
    {
        bp_layer_act (plane->m_layers[i]);
        bp_layer_act_weight (plane->m_layers[i], lr);
    }
}

void bp_neuron_act_weight (Neuron *n, gdouble lr)
{
    for (int i = 0; i < n->m_size_bp; i ++)
    {
        *n->m_weights_bp[i] += (lr * *n->m_inputs_bp[i] * n->m_output);
    }
}

void bp_neuron_act (Neuron *n)
{
    gdouble sum = 0.0f;
    for (int i = 0; i < n->m_size_bp; i ++)
    {
        sum += (*n->m_inputs_bp[i] * (*n->m_weights_bp[i]));
    }
    n->m_delta = sum * n->m_dact_func (n->m_output);
}


void bp_layer_act_weight (Layer *l, gdouble lr)
{
    for (int i = 0; i < layer_size (l); i ++)
    {
        bp_neuron_act_weight (layer_neuron (l, i), lr);
    }
}

void bp_layer_act (Layer *l)
{
    for (int i = 0; i < layer_size (l); i ++)
    {
        bp_neuron_act (layer_neuron (l, i));
    }
}

void load_data (PlaneBin *bin, gdouble *inputs)
{
    for (int i = 0; i < bin->m_size; i ++)
    {
#ifdef DEBUG_DATA
        DBG ("loading data: ");
#endif
        Layer *il = bin->m_plane[i]->m_layers[0];
        for (int j = 0; j < layer_raw_size (il); j ++)
        {
            layer_neuron (il, j)->m_output = inputs[j];
#ifdef DEBUG_DATA
            DBG ("%g ", layer_neuron (il, j)->m_output);
#endif
        }
        NL;
    }
}


void bp_plane_bin_act (PlaneBin *pb, gdouble *exp, gdouble lr)
{
    pb->m_lr = lr;
    for (int i = 0; i < pb->m_size; i ++)
    {
        bp_plane_act (pb->m_plane[i], exp, lr);
    }
}

void fwd_plane_act (Plane *plane)
{
    /*
     * bypassing input layer
     */
    for (int i = 1; i < plane->m_ln; i ++)
    {
        fwd_layer_act (plane->m_layers[i]);
    }
}

void fwd_plane_bin_act (PlaneBin *pb)
{
    for (int i = 0; i < pb->m_size; i ++)
    {
        fwd_plane_act (pb->m_plane[i]);
    }
}

gboolean AlmostEqualRelative(gdouble A, gdouble B,
                         gdouble maxRelDiff)
{
    // Calculate the difference.
    float diff = fabs(A - B);
    A = fabs(A);
    B = fabs(B);
    // Find the largest
    float largest = (B > A) ? B : A;
 
    if (diff <= (largest * maxRelDiff))
        return TRUE;
    return FALSE;
}

