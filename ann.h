#ifndef _ANN_H_
#define _ANN_H_

#include <glib.h>

struct _Neuron;
struct _Layer;
struct _OuputLayer;
struct _Plane;
struct _PlaneBin;
struct _LayerOutput;
struct _ArraySize;

typedef gdouble (*actviation_func) (gdouble x);
typedef gdouble (*delta_actviation_func) (gdouble x);

extern gchar * const sep;
extern gchar * const ln_input;
extern gchar * const ln_hidden;
extern gchar * const ln_output;

typedef enum
{
    mask_dw = (1 << 0),        //weight
    mask_di = (1 << 1),        //input
    mask_do = (1 << 2),        //output
    mask_dd = (1 << 3),        //delta
    mask_db = (1 << 4),        //basic info
    mask_da = 0x1F
} dump_mask;

typedef enum
{
    df_fwd,
    df_bwd
} data_flow;

typedef enum
{
    lt_input,
    lt_hidden,
    lt_output
} layer_type;

typedef struct _ArraySize
{
    guint m_x;
    guint m_y;
} ArraySize;

/*
 * each node has a set of weights, which randomly assigned
 * but inputs are shared among nodes
 * output of preivous layer is the input for the next
 * again it is shared among nodes of the next
 */
typedef struct _Neuron
{
    //for fwd 
    gdouble **m_inputs;
    gdouble **m_weights;
    gdouble m_output;
    guint m_size;
    actviation_func m_act_func;
    //for backward
    gdouble **m_inputs_bp;
    gdouble **m_weights_bp;
    guint m_size_bp;
    gdouble m_delta;
    delta_actviation_func m_dact_func;
    //for all
    guint m_id;
    guint m_layer_id;
} Neuron; 

typedef struct _Layer
{
    Neuron **m_neurons;
    guint m_size;
    guint m_scale;
    guint m_id;
    guint m_plane_id;
    gboolean m_with_bias;
    layer_type m_lt;
} Layer;

typedef struct _LayerOutput
{
    gdouble **m_output;
    guint m_size;
} LayerOutput;

typedef struct _Plane
{
    Layer **m_layers;
    guint m_ln;
    gdouble ***m_weights;
    ArraySize *m_dims;
    GAsyncQueue **m_q_layers;
    guint m_id;
} Plane;

typedef struct _PlaneBin
{
    Plane **m_plane;
    guint m_size;
    gdouble ***m_weights;
    guint m_w_num;
    ArraySize *m_dims;
} PlaneBin;

/*
 * helper functions 
 */

gchar *layer_name (layer_type lt);
guint layer_size (Layer *l);
Neuron *layer_neuron (Layer *l, guint i);
gdouble **layer_neuron_w_bp (Layer *l, guint i, guint j);
gdouble **layer_neuron_w (Layer *l, guint i, guint j);
gdouble layer_neuron_weight (Layer *l, guint i, guint j);
double layer_neuron_weight_bp (Layer *l, guint i, guint j);
gsize layer_input_scale (Layer *l);
gdouble **neuron_input (Neuron *n, guint i);
gsize layer_raw_size (Layer *l);
gsize layer_start_neuron (Layer *l);


gboolean AlmostEqualRelative(gdouble A, gdouble B, gdouble maxRelDiff);

#define FLT_EPSILON __FLT_EPSILON__
#define flt_equal(a, b) AlmostEqualRelative (a, b, FLT_EPSILON)

/*
 * leveraging glib GArray for GArray
 */


gdouble act_sigmoid (gdouble x);
gdouble dact_sigmoid (gdouble x);
//gdouble act_tanh (gdouble x);
gdouble act_relu (gdouble x);
gdouble dact_relu (gdouble x);
gdouble act_threshold (gdouble x);
gdouble dact_threshold (gdouble x);

/*
 * typedef void (*GDestroyNotify) (gpointer data);
 */

void free_array (gdouble **arr, guint s);
void free_neuron (Neuron *n);
void free_layer (Layer *l);
void free_layer_output (LayerOutput *lo);
void free_plane (Plane *plane);
void free_plane_bin (PlaneBin *pb);

Neuron *construct_neurons (guint n, guint id);
Layer *construct_layer (guint n, guint m, guint id, gboolean with_bias, layer_type lt);

LayerOutput *construct_layer_output (guint n);
Plane *construct_plane (guint in, guint hn, guint on, guint hl, guint id);
PlaneBin *construct_plane_bin (guint pn, guint in, guint hn, guint on, guint hl);

void init_plane_bp (Plane *fwd);

gboolean link_plane_layers (Plane *plane);
gboolean link_layers (Layer *l1, Layer *l2);

void fwd_neuron_act (Neuron *n);
void fwd_layer_act (Layer *plane);
void fwd_plane_act (Plane *plane);
void fwd_plane_bin_act (PlaneBin *pb);

void bp_layer_act (Layer *l);
void bp_neuron_act (Neuron *n);
void bp_layer_act_bp (Layer *l1, Layer *l2);
void bp_layer_act_exp (Layer *l, gdouble *exp);
void bp_plane_act (Plane *plane, gdouble *exp, gdouble lr);
void bp_plane_bin_act (PlaneBin *pb, gdouble *exp, gdouble lr);

void bp_neuron_act_weight (Neuron *n, gdouble lr);
void bp_layer_act_weight (Layer *l, gdouble lr);
/*
 * To be obseleted
 */
void bp_layer_weight (Layer *l1, Layer *l2, gdouble lr);


void dump_plane_bin (PlaneBin *pb, guint dump_bits);
void dump_weights (gdouble ***w, ArraySize *as, guint ln);
void dump_plane (Plane *pb, guint dump_bits);
void dump_layer (Layer *l, guint dump_bits);
void dump_neuron (Neuron *n, guint dump_bits);

void set_plane_func (Plane *plane, actviation_func func, delta_actviation_func dfunc);
void set_layer_func (Layer* l, actviation_func func, delta_actviation_func dfunc);
void init_weights_randomly (PlaneBin *pb);
void assign_weights_to_neuron (Plane *plane);
void assign_weights_to_layer (Layer *l, ArraySize *as, gdouble **w);
void assign_weights_to_layer_bp (Layer *l, ArraySize *as, gdouble **w);


void load_data (PlaneBin *bin, gdouble *inputs);

#ifdef DEBUG 
#define DBG(...) g_printerr (__VA_ARGS__)
#define DL DBG ("%d\n", __LINE__)
#define NL DBG ("\n")
#define SEP DBG ("%s\n", sep)
#else
#define DBG(...)
#define DL 
#define NL
#define SEP
#endif

#endif
