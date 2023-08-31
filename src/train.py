from settings import *

# initialize timers
timer, tot_exec_timer, io_timer, train_timer = lb.utils.init_timer(runtype='training')

# start total timer
timer.start(tot_exec_timer)

# initialize logger
logging.basicConfig(format="[%(asctime)s] %(levelname)s : %(message)s", level=logging.DEBUG, filename=LOG_FILE, datefmt='%Y-%m-%d %H:%M:%S')

# log 
logging.debug(f'Logger initialized')

# copy configuration file into the run folder
shutil.copyfile(args.config, join(RUN_DIR, 'configuration.toml'))

# download dataset
# url = "https://drive.google.com/drive/folders/15DEq33MmtRvIpe2bNCg44lnfvEiHcPaf"
# gdown.download_folder(url=url, quiet=False, output=config.dirs.data)

# define driver information and tensor coder for TFRecord decoding
drivers_info = [lb.record.io.DriverInfo(vars=config.data.drivers, shape=config.data.drivers_shape), lb.record.io.DriverInfo(vars=config.data.targets, shape=config.data.targets_shape)]
tensor_coder = lb.record.io.TensorCoder(drivers_info=drivers_info)

# log
logging.debug(f'Drivers info and tensor coder initialized')

# time
timer.start(io_timer)

# get data filenames for training and validation
train_cyc_fnames, train_rnd_fnames, train_adj_fnames = get_data_filenames(dataset_dir=config.dirs.dataset, years=lb.macros.TRAIN_YEARS, patch_type=config.data.patch_type, runtype='training')
valid_cyc_fnames, valid_rnd_fnames, valid_adj_fnames = get_data_filenames(dataset_dir=config.dirs.dataset, years=lb.macros.VALID_YEARS, patch_type=config.data.patch_type, runtype='training')

# log
logging.debug(f'Training and validation filenames loaded. There are {len(train_cyc_fnames)+len(train_rnd_fnames)+len(train_adj_fnames)} files for training and {len(valid_cyc_fnames)+len(valid_rnd_fnames)+len(valid_adj_fnames)} files for validation')

# compute the scaler
x_scaler = compute_tf_scaler(cyc_fnames=train_cyc_fnames, rnd_fnames=train_rnd_fnames, adj_fnames=train_adj_fnames, tensor_coder=tensor_coder, scaler_file=X_SCALER_FILE, config=config)

# time
timer.stop(io_timer)

# setup mirrored strategy
mirrored_strategy = tf.distribute.MirroredStrategy()

# log
logging.debug(f'Mirrored strategy init. Replicas in sync: {mirrored_strategy.num_replicas_in_sync}')

# time
timer.start(io_timer)

# load training and validation datasets
train_dataset, train_steps = load_dataset(cyc_fnames=train_cyc_fnames, rnd_fnames=train_rnd_fnames, adj_fnames=train_adj_fnames, x_scaler=x_scaler, tensor_coder=tensor_coder, config=config)
valid_dataset, valid_steps = load_dataset(cyc_fnames=valid_cyc_fnames, rnd_fnames=valid_rnd_fnames, adj_fnames=valid_adj_fnames, x_scaler=x_scaler, tensor_coder=tensor_coder, config=config)

# log
logging.debug(f'Datasets retrieved')

# distribute datasets
dist_train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
dist_valid_dataset = mirrored_strategy.experimental_distribute_dataset(valid_dataset)

# time
timer.stop(io_timer)

# log
logging.debug(f'Datasets distributed')

# time
timer.start(train_timer)

model, loss, optimizer, metrics, callbacks = load_compiled_model(config, mirrored_strategy)

# log
logging.debug(f'Starting training')

# fit the model with the dataset
history = lb.utils.model_fit(model=model, train_dataset=dist_train_dataset, valid_dataset=dist_valid_dataset, train_steps=train_steps, valid_steps=valid_steps, epochs=config.training.epochs, callbacks=callbacks)

# log
logging.debug(f'Training completed')

# time
timer.stop(train_timer)

# time
timer.start(io_timer)

# save the model to disk
lb.utils.save_model(model=model, dst=LAST_MODEL)

# log
logging.debug(f'Last model saved')

# time
timer.stop(io_timer)

# save training and validation training times
pd.DataFrame(data={"Training exec time" : [timer.exec_times[train_timer]], "I/O exec time" : [timer.exec_times[io_timer]], "Total Execution Time" : [timer.exec_times[tot_exec_timer]]}).to_csv(TRAINVAL_TIME_CSV)

# log
logging.debug(f'Process completed')
