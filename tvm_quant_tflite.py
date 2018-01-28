


import os

import datetime 
import tflite.Model
import numpy as np

import tvm 
from tvm import te
from tvm import autotvm
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm.autotvm.tuner import GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime

def extract(path):
    import tarfile
    if path.endswith("tgz") or path.endswith("gz"):
        dir_path = os.path.dirname(path)
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError('Could not decompress the file: ' + path)


def get_quantized_network(quant_model_url, model_name, batch_size=1, input_tensor='input',
                            input_shape=(224, 224, 3), input_dtype='uint8'):
    
    data_shape = (batch_size,) + input_shape

    model_path = download_testdata(quant_model_url, model_name + ".tgz", module=['tf', 'official'])
    model_dir = os.path.dirname(model_path)
    extract(model_path)

    tflite_model_file = os.path.join(model_dir, model_name + ".tflite")
    tflite_model_buf = open(tflite_model_file, "rb").read()
    
    # Get TFLite model from buffer
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    # TFLite framework to Relay parser - Default layout is NHWC
    mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict={input_tensor: data_shape},
                                         dtype_dict={input_tensor: input_dtype})

   # Convert the layout to NCHW
    # RemoveUnunsedFunctions is used to clean up the graph.
    seq = relay.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                  relay.transform.ConvertLayout('NCHW')])
    with relay.transform.PassContext(opt_level=3):
        transformed_mod = seq(mod)
        
    return transformed_mod, params, data_shape

def tune_tasks(tasks,
                 measure_option,
                 tuner='gridsearch',
                 early_stopping=None,
                 log_filename='tuning.log'):

    
    tmp_log_file = log_filename + ".tmp"
    
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)
    
    for i, task in enumerate(tasks):
        
        print(task)
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)
        
        # do tuning
        #n_trial=len(task.config_space)
        n_trial = 2
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])
    
    # pick best records to a cache file 
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def tune(tuning_opt, target, model_url, model_name, 
        batch_size, input_tensor, input_shape=(224, 224, 3), input_dtype='uint8', need_tune=True):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, data_shape = get_quantized_network(model_url, model_name, batch_size, 
        input_tensor, input_shape=input_shape, input_dtype=input_dtype)

    if need_tune:
        
        tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                                params=params,
                                                ops=(relay.op.get("nn.conv2d"),))
        print("Get the tasks...%d" % len(tasks))
        # run tuning tasks
        #tune_kernels(tasks, **tuning_opt)
        print('Tuning...')
        tune_tasks(tasks, **tuning_opt)
        
    return mod, params, data_shape

def evaluate(log_file, mod, params, target, input_tensor='input', data_shape=(1, 224, 224, 3), input_dtype='uint8'):
    # compile kernels with graph-level best records
    #with autotvm.apply_history_best(log_file):
    with open(log_file, 'r') as f:
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # upload parameters to device
        ctx = tvm.cpu()
        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(input_dtype))
        module = runtime.create(graph, lib, ctx)
        module.set_input(input_tensor, data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

def main():

    target = tvm.target.arm_cpu()
    
    batch_size = 1
    dtype = 'uint8'
    
    quant_model_url = "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz"
    model_name = "mobilenet_v1_1.0_224_quant"
    log_file = "%s.log" % model_name
    
    
    input_tensor = "input"

    tuning_option = {
    'log_filename': log_file,
    'tuner': 'random',
    'early_stopping': 800,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=10, repeat=1,
                                   min_repeat_ms=1000),
    ),}

    mod, params, data_shape = tune(tuning_option, target, quant_model_url, model_name, batch_size, 
        input_tensor, need_tune=False)
    
    evaluate(log_file, mod, params, target, input_tensor, data_shape, input_dtype=dtype)

if __name__ =='__main__':
    main()

    