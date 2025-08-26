#!/usr/bin/env python

def voriticity()

    import os
    from pyophidia import Client, Workflow, Experiment, Cube
    home_dir = os.path.expanduser('~')
    
    # Input parameters
    input_folder = home_dir + "/work/vorticity/input"
    output_folder = home_dir + "/work/vorticity/output"
    output_variable1 = "msl"
    output_variable2 = "vo_850"
    lat_range = "0:70"
    lon_range = "100:320"
    plev_name = "plev" # plev or lev (for MPI-ESM1-2-HR)
    plev_type = "float" # double or float (for CNRM-CM6-1-HR)
    regrid_script = home_dir + "/work/vorticity/regrid.sh"
    merge_script = home_dir + "/work/vorticity/merge.sh"
    new_grid = "r880x280"
    display = True
    number_of_files = "1"
    query_on_files = "*_201*.nc"
    init_script = home_dir + "/work/vorticity/linkinput.sh"
    #init_script = home_dir + "/work/vorticity/linkinput-CNRM.sh" # (for CNRM-CM6-1-HR)
    
    # Costants
    hosts = "1"
    cores = "1"
    threads = "1"
    partition = "partition"
    container = "wind"
    lon_file = home_dir + "/data/CMIP6/HighResMIP/CMCC/CMCC-CM2-VHR4/highres-future/r1i1p1f1/6hrPlevPt/psl/gn/v20190509/psl_6hrPlevPt_CMCC-CM2-VHR4_highres-future_r1i1p1f1_gn_201501010000-201501311800.nc"
    #lon_file = home_dir + "/data/CMIP6/HighResMIP/EC-Earth-Consortium/EC-Earth3P-HR/highres-future/r1i1p2f1/6hrPlevPt/psl/gr/latest/psl_6hrPlevPt_EC-Earth3P-HR_highres-future_r1i1p2f1_gr_201501010000-201512311800.nc"
    #lon_file = home_dir + "/data/CMIP6/HighResMIP/MPI-M/MPI-ESM1-2-HR/highres-future/r1i1p1f1/6hrPlevPt/psl/gn/latest/psl_6hrPlevPt_MPI-ESM1-2-HR_highres-future_r1i1p1f1_gn_201501010556-201512312356.nc"
    #lon_file = home_dir + "/data/CMIP6/HighResMIP/CNRM-CERFACS/CNRM-CM6-1-HR/highres-future/r1i1p1f2/6hrPlevPt/psl/gr/latest/psl_6hrPlevPt_CNRM-CM6-1-HR_highres-future_r1i1p1f2_gr_201501010600-202001010000.nc"
    core_limit = "20"
    time_part1 = lon_file.index(lon_file.split("_")[-1]) - 2
    time_part2 = time_part1 + 13
    eq_ray = "6378137" # m
    pol_ray = "6356702" # m
    q_ray = "6372797" # m
    
    cli = Client(read_env = True)
    Cube.setclient(cli)
    
    Cube.cluster(action = 'deploy', host_partition = partition, nhost = hosts, exec_mode = 'async')
    
    exp = Experiment(name="Vorticity",
                    author="CMCC",
                    abstract="This workflow evaluate the wind vorticity dv/dlon - du/dlat given the wind components u and v using PyOphidia",
                    exec_mode="sync",
                    host_partition=partition,
                    nthreads=threads,
                    on_exit="oph_fastdelete",
                    ncores=cores)
    
    t0 = exp.newTask(name="Initialization",
                    operator="oph_script",
                    arguments={"script": init_script, "args": number_of_files + " " + query_on_files})
    
    t1 = exp.newTask(name="Create a work container",
                    operator="oph_createcontainer",
                    on_error="skip",
                    arguments={"container": container, "dim": "time|" + plev_name + "|lat|lon"},
                    dependencies={t0:''})
    
    t2 = exp.newTask(name="Get latitude",
                    operator="oph_importnc2",
                    arguments={"imp_dim": "lat", "measure": "lat", "input": lon_file, "container": container, "nfrag": "1", "subset_dims": "lat", "subset_filter": lat_range, "subset_type": "coord"},
                    dependencies={t1:''})
    
    t3a = exp.newTask(name="Change unit of latitude",
                    operator="oph_apply",
                    arguments={"query": "oph_matheval('oph_double','oph_float',measure,'pi*x/180')"},
    #                arguments={"query": "oph_matheval(measure,'cos(pi*x/180)')", "measure_type": "auto"}, # Valid for EC-Earth
                    dependencies={t2:'cube'})
    
    t3b = exp.newTask(name="Evaluate ray factor",
                    operator="oph_apply",
                    arguments={"query": "oph_matheval(measure,'"+eq_ray+"*"+pol_ray+"/sqrt(("+eq_ray+"*sin(x))^2+("+pol_ray+"*cos(x))^2)*pi/180')", "measure_type": "auto"},
                    dependencies={t3a:'cube'})
    
    t3c = exp.newTask(name="Evaluate latitude factor",
                    operator="oph_apply",
                    arguments={"query": "oph_matheval(measure,'"+eq_ray+"*"+pol_ray+"/sqrt(("+eq_ray+"*tan(x))^2+("+pol_ray+")^2)*pi/180')", "measure_type": "auto"},
                    dependencies={t3a:'cube'})
    
    t4 = exp.newTask(name="Iterate on time",
                    operator="oph_for",
                    arguments={"parallel": "yes", "key": "source", "input": "[" + input_folder + "/ua_*.nc]"},
                    dependencies={t0:''})
    
    tcc = exp.newTask(name="Create a temporary container",
                    operator="oph_createcontainer",
                    on_error="skip",
                    arguments={"container": container + "_&{source}", "dim": "time|" + plev_name + "|lat|lon"},
                    dependencies={t4:''})
    
    tp1 = exp.newTask(name="Import PSL",
                    operator="oph_importnc2",
                    arguments={"imp_dim": "lat", "measure": "psl", "input": input_folder + "/psl_@{source_file+3}", "container": container + "_&{source}", "nfrag": threads, "subset_dims": "lat|lon", "subset_filter": lat_range + "|" + lon_range, "subset_type": "coord"},
    #                operator="oph_importncs", # Valid for CNRM
    #                arguments={"imp_dim": "lat", "measure": "psl", "input": input_folder + "/psl_*.nc", "container": container + "_&{source}", "nfrag": threads, "subset_dims": "time|lat|lon", "subset_filter": "@{source+" + str(time_part1) + "-17}0_@{source+" + str(time_part2) + "-4}9|" + lat_range + "|" + lon_range, "subset_type": "coord"}, # Valid for CNRM
                    dependencies={tcc:''})
    
    tp2 = exp.newTask(name="Rename measure",
                    operator="oph_apply",
                    arguments={"measure": output_variable1},
                    dependencies={tp1:'cube'})
    
    tp3 = exp.newTask(name="Export PSL",
                    operator="oph_exportnc2",
                    arguments={"output": output_folder + "/psl_@{source_file+3}"},
                    dependencies={tp2:'cube'})
    
    tp4 = exp.newTask(name="Regrid PSL",
                    operator="oph_generic",
                    arguments={"command": regrid_script, "output": output_folder + "/regridded_psl_@{source_file+3}", "args": lat_range + " " + lon_range + " " + new_grid},
                    dependencies={tp3:'input'})
    
    tu1 = exp.newTask(name="Import U",
                    operator="oph_importnc2",
                    arguments={"imp_dim": "lat", "measure": "ua", "input": input_folder + "/ua_@{source_file+3}", "container": container + "_&{source}", "nfrag": threads, "subset_dims": plev_name + "|lat|lon", "subset_filter": "85000|" + lat_range + "|" + lon_range, "subset_type": "coord"},
                    dependencies={tcc:''})
    
    tu1b = exp.newTask(name="Put ray factor",
                    operator="oph_intercube", 
                    arguments={ "operation": "div", "cube2_is_array": "yes" },
                    dependencies={tu1:'cube', t3b:'cube2'})
    
    tu2 = exp.newTask(name="Evaluate du/dlat",
                    operator="oph_apply",
                    arguments={"query": "oph_gsl_spline(measure,dimension,dimension,1)", "measure_type": "auto"},
                    dependencies={tu1b:'cube'})
    
    tu3 = exp.newTask(name="Rollup U",
                    operator="oph_rollup",
                    arguments={},
                    dependencies={tu2:'cube'})
    
    tu4 = exp.newTask(name="Permute U",
                    operator="oph_permute",
                    arguments={"dim_pos": "2,1"},
                    dependencies={tu3:'cube'})
    
    tv1 = exp.newTask(name="Import V",
                    operator="oph_importnc2",
                    arguments={"imp_dim": "lon", "measure": "va", "input": input_folder + "/va_@{source_file+3}", "container": container + "_&{source}", "nfrag": threads, "subset_dims": plev_name + "|lat|lon", "subset_filter": "85000|" + lat_range + "|" + lon_range, "subset_type": "coord"},
                    dependencies={tcc:''})
    
    tv2 = exp.newTask(name="Evaluate dv/dlon",
                    operator="oph_apply",
                    arguments={"query": "oph_gsl_spline(measure,dimension,dimension,1)", "measure_type": "auto"},
                    dependencies={tv1:'cube'})
    
    tv3 = exp.newTask(name="Rollup V",
                    operator="oph_rollup",
                    arguments={},
                    dependencies={tv2:'cube'})
    
    tv4 = exp.newTask(name="Put latitude factor",
                    operator="oph_intercube", 
                    arguments={ "operation": "div", "cube2_is_array": "yes", "extension_type": "interlace" },
                    dependencies={tv3:'cube', t3c:'cube2'})
    
    t5 = exp.newTask(name="Evaluate vorticity",
                    operator="oph_intercube", 
                    arguments={ "measure": output_variable2 },
                    dependencies={tv4:'cube', tu4:'cube2'})
    
    t6 = exp.newTask(name="Delete metadata",
                    operator="oph_metadata", 
                    arguments={ "mode": "delete", "variable": output_variable2, "metadata_key": "comment" },
                    dependencies={t5:'cube'})
    
    t7 = exp.newTask(name="Update metadata",
                    operator="oph_metadata", 
                    arguments={ "mode": "update", "variable": output_variable2, "metadata_key": "standard_name|long_name|units", "metadata_value": "atmosphere_relative_vorticity|Vorticity (relative)|s**-1", "force": "yes" },
                    dependencies={t6:'cube'})
    
    t8 = exp.newTask(name="Export VO",
                    operator="oph_exportnc2",
                    arguments={"output": output_folder + "/vo_@{source_file+3}"},
                    dependencies={t7:'cube'})
    
    t9 = exp.newTask(name="Regrid VO",
                    operator="oph_generic",
                    arguments={"command": regrid_script, "output": output_folder + "/regridded_vo_@{source_file+3}", "args": lat_range + " " + lon_range + " " + new_grid},
                    dependencies={t8:'input'})
    
    t10 = exp.newTask(name="Merge outputs",
                    operator="oph_generic",
                    arguments={"command": merge_script, "output": output_folder + "/both_@{source_file+3}", "args": output_variable1},
                    dependencies={tp4:'input', t9:'input'})
    
    tcd = exp.newTask(name="Destroy the temporary container",
                    operator="oph_deletecontainer",
                    on_error="skip",
                    arguments={"container": container + "_&{source}", "force": "yes"},
                    dependencies={t10:''})
    
    t19 = exp.newTask(name="End iteration",
                    operator="oph_endfor",
                    arguments={},
                    dependencies={tcd:''})
    
    tcd = exp.newTask(name="Destroy the work container",
                    operator="oph_deletecontainer",
                    on_error="skip",
                    arguments={"container": container, "force": "yes"},
                    dependencies={t19:''})

    Workflow.setclient(cli)
    wf = Workflow(exp)
    wf.submit(exec_mode = "sync")
    
    wf.monitor(iterative = False, display = display, save = True)
    json_prov = wf.build_provenance("Vorticity", output_format = "json", display = display)
    
    Cube.cluster(action = 'undeploy', host_partition = partition, exec_mode = 'sync')
    
    cli.submit("oph_delete cube=[*]", display = False)
    
    print("Workflow completed")

if __name__ == "__main__":
    vorticity()

