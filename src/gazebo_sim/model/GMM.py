import numpy as np

from dcgmm.model        import DCGMM
from dcgmm.layer        import GMM_Layer, Folding_Layer, Readout_Layer, Input_Layer, Reshape_Layer, Concatenate_Layer
from dcgmm.callback     import Log_Protos, Set_Model_Params, Early_Stop

from cl_experiment.callback         import Log_Metrics
from cl_experiment.utils            import change_loglevel
from cl_experiment.checkpointing    import Manager as Checkpoint_Manager

import math ;
from gazebo_sim.utils.logger import log

def build_model(name, input_dims, output_size, config, ro_loss="mean_squred_error"):
    """ returns a GMM keras model instance. """
    log("GMM BUILD MODEL!!", config.qgmm_lambda_W, config.qgmm_reset_somSigma) ;
    input_dims = list(input_dims)
    inputs = Input_Layer(
        layer_name='L0_INPUT',
        prefix='L0_',
        shape=input_dims).create_obj()
    fold = Folding_Layer(
        layer_name="L1_FOLD",
        prefix='L1_',
        patch_width=-1,
        patch_height=-1,
        stride_x=1,
        stride_y=1,
        sampling_batch_size=config.train_batch_size,
    )(inputs)
    gmm_layer = GMM_Layer(
        layer_name='L2_GMM',
        prefix='L2_',
        input_layer=1,
        L2_K=config.qgmm_K,
        L2_conv_mode='yes',
        L2_lambda_sigma=0.1,
        L2_lambda_mu=1.,
        L2_lambda_pi=0.,
        L2_eps_0=0.011,  # 0.0051
        L2_eps_inf=0.01,  # 0.005
        L2_sigmaUpperBound=10., 
        L2_somSigma_sampling=config.qgmm_somSigma_sampling,
        L2_sampling_batch_size=config.train_batch_size,
        L2_sampling_divisor=20.,
        L2_sampling_I=-1,
        L2_sampling_S=1,
        L2_sampling_P=1.,
        L2_reset_factor=config.qgmm_reset_somSigma,
        L2_alpha=config.qgmm_alpha,
        L2_gamma=config.qgmm_gamma,
        L2_regularizer="NewReg", 
        L2_somSigma_0=1.41*math.sqrt(config.qgmm_K) * 0.4, 
        L2_delta=0.3)(fold)
    ro_layer = Readout_Layer(
        layer_name='L3_READOUT',
        prefix='L3_',
        input_layer=2,
        L3_num_classes=output_size,
        L3_loss_function=ro_loss,
        L3_sampling_batch_size=config.train_batch_size,
        L3_regEps=config.qgmm_regEps,  # 0.05, 0.01
        L3_lambda_W=config.qgmm_lambda_W,
        L3_lambda_b=config.qgmm_lambda_b,
    )(gmm_layer)

    exp_id = config.exp_id
    root_dir = config.root_dir

    log_path = config.root_dir ; # TODO: anpassen!

    
    # ---------- MODEL
    
    model = DCGMM(
        inputs=inputs, outputs=ro_layer, log_level='INFO', name=name,
        architecture='QGMM', exp_id=config.exp_id, wandb_active='no',
        batch_size=config.train_batch_size,
        test_batch_size=config.train_batch_size,
        sampling_batch_size=config.train_batch_size,
        ro_patience=-1, 
        vis_path=log_path,
    )
    
    model.compile(run_eagerly=True)
    model.build(input_shape=input_dims)

    model_str = []
    model.summary(print_fn=lambda x: model_str.append(x))
    model_summary = '\n'.join(model_str) 

    # ---------- LOAD A PRE-TRAINED GMM CHECKPOINT (warm-start)
    if config.qgmm_load_ckpt != "no":
        log("LOADING!!") ;
        model.load_weights(config.qgmm_load_ckpt) ;
        #model.reset() # reset som_Sigma state to reset_factor value
        log("CLEARW!!!!!!!") ;
        nW = np.zeros(model.layers[3].W.shape)
        model.layers[3].W.assign(nW) ;
        model.layers[3].b.assign(model.layers[3].b*0.0) ;
        
    return model
