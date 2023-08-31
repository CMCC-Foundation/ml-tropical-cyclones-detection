import tensorflow as tf
import numpy as np
import joblib
import time
import os



def seed_everything(seed):
    # python seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # numpy seed
    np.random.seed(seed)
    # tensorflow seed
    tf.random.set_seed(seed)
    print(f'Execution seeded successfully with seed {seed}')



class Timer():

    def __init__(self, timers=['tot_exec_elapsed_time', 'io_elapsed_time', 'training_elapsed_time']):
        # initialize execution times data structure
        self.exec_times = {}
        self.partials = {}
        for t in timers:
            self.exec_times.update({t:0})
            self.partials.update({t:0})

    def start(self, timer):
        # update partial timers to start counting
        self.partials.update({timer:-time.time()})
    
    def stop(self, timer):
        # add ( stop - start ) time to global execution time
        self.exec_times[timer] += self.partials[timer] + time.time()
        # reset partial
        self.partials[timer] = 0



def init_timer(runtype):
    """
    Initializes the timer for ML workflows. Time is divided into:
    1. total execution time
    2. io execution time
    3. training/test execution time
    
    """
    # define timer names
    tot_exec_timer = 'tot_exec_elapsed_time'
    io_timer = 'io_elapsed_time'
    if runtype == 'training':
        exec_timer = 'training_elapsed_time'
    elif runtype == 'test':
        exec_timer = 'inference_elapsed_time'
    # running time setup
    timer = Timer(timers=[tot_exec_timer, io_timer, exec_timer])
    return timer, tot_exec_timer, io_timer, exec_timer



def joblib_save(dst, data=None, **kwargs):
    """
    Saves 'data' or 'kwargs' to the destination file 'dst'.

    """
    try:
        if data:
            joblib.dump(data, dst)
        else:
            joblib.dump(kwargs, dst)
    except Exception as e:
        print(e)
        return



def joblib_load(src):
    """
    Loads from 'src' filepath the data.
    
    """
    try:
        data = joblib.load(filename=src)
    except Exception as e:
        print(e)
        return
    return data



def save_model(model, dst):
    try:
        model.save(dst, save_format='tf', include_optimizer=True)
    except:
        model.save(dst, save_format='tf', include_optimizer=False) # valutare se lasciare questo



def model_fit(model, train_dataset, valid_dataset, train_steps, valid_steps, epochs, callbacks):
    # fit model on the dataset
    history = model.fit(
        train_dataset, 
        validation_data=valid_dataset, 
        steps_per_epoch=train_steps, 
        validation_steps=valid_steps, 
        epochs=epochs, 
        callbacks=callbacks)

    return history
