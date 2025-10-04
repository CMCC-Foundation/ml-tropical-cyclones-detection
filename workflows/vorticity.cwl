#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: Workflow
label: This workflow evaluate the wind vorticity dv/dlon - du/dlat given the wind components u and v using PyOphidia

requirements:
  MultipleInputFeatureRequirement: {}
  InlineJavascriptRequirement: {}
  StepInputExpressionRequirement: {}

inputs:
  inputexperiment:
    type: File?
  nthreads: int
  lon_file: string
  container: string
  lat_range: string
  space_range: string
  number_of_files: int
  query_on_files: string
  output_variable1: string
  output_variable2: string

outputs:
  outputexperiment:
    type: File
    outputSource: Destroy_the_work_container/experiment

steps:
  Initialization:
    run: tasks/script.cwl
    in:
      experiment: inputexperiment
      name: 
        default: "Initialization"
      script: 
        default: "/home/jovyan/work/vorticity/linkinput.sh"
      args:
        valueFrom: $(inputs.number_of_files) $(inputs.query_on_files)
      number_of_files: number_of_files
      query_on_files: query_on_files
    out: [experiment]

  Create_a_work_container:
    run: tasks/createcontainer.cwl
    in:
      experiment: Initialization/experiment
      name: 
        default: "Create a work container"
      container: container
      dim:
        default: "time|plev|lat|lon"
      on_error:
        default: "skip"
    out: [experiment]

  Get_latitude:
    run: tasks/importnc2.cwl
    in:
      experiment: Create_a_work_container/experiment
      name:
        default: "Get latitude"
      src_path: lon_file
      measure:
        default: "lat"
      imp_dim:
        default: "lat"
      container: container
      nfrag:
        default: 1
      subset_dims:
        default: "lat"
      subset_filter: lat_range
      subset_type:
        default: "coord"
    out: [experiment]
  Change_unit_of_latitude:
    run: tasks/apply.cwl
    in:
      experiment: Get_latitude/experiment
      name:
        default: "Change unit of latitude"
      query:
        default: "oph_matheval('oph_double','oph_float',measure,'pi*x/180')"
    out: [experiment]
  Evaluate_radius_factor:
    run: tasks/apply.cwl
    in:
      experiment: Change_unit_of_latitude/experiment
      name:
        default: "Evaluate radius factor"
      query:
        default: "oph_matheval(measure,'sqrt(((6378137^2*cos(x))^2+(6356752^2*sin(x))^2)/((6378137*cos(x))^2+(6356752*sin(x))^2))*pi/180')"
      measure_type:
        default: "auto"
    out: [experiment]
  Evaluate_latitude_factor:
    run: tasks/apply.cwl
    in:
      experiment: Change_unit_of_latitude/experiment
      name:
        default: "Evaluate latitude factor"
      query:
        default: "oph_matheval(measure,'sqrt(((6378137^2*cos(x))^2+(6356752^2*sin(x))^2)/((6378137*cos(x))^2+(6356752*sin(x))^2))*pi/180*cos(x)')"
      measure_type:
        default: "auto"
    out: [experiment]

  Iterate_on_time:
    run: tasks/for.cwl
    in:
      experiment: Create_a_work_container/experiment
      name:
        default: "Iterate on time"
      parallel:
        default: "yes"
      key:
        default: "source"
      input:
        default: "[/home/jovyan/work/vorticity/input/psl_*.nc]"
    out: [experiment]

  Import_PSL:
    run: tasks/importnc2.cwl
    in:
      experiment: Iterate_on_time/experiment
      name:
        default: "Import PSL"
      src_path:
        default: "@source"
      measure:
        default: "psl"
      imp_dim:
        default: "lat"
      container: container
      subset_dims:
        default: "lat|lon"
      subset_filter: space_range
      subset_type:
        default: "coord"
      nfrag: nthreads
      nthreads: nthreads
    out: [experiment]
  Rename_measure:
    run: tasks/apply.cwl
    in:
      experiment: Import_PSL/experiment
      name:
        default: "Rename measure"
      measure: output_variable1
    out: [experiment]
  Export_PSL:
    run: tasks/exportnc2.cwl
    in:
      experiment: Rename_measure/experiment
      name:
        default: "Export PSL"
      output:
        default: "/home/jovyan/work/vorticity/output/@{source_file}"
    out: [experiment]
  Regrid_PSL:
    run: tasks/generic.cwl
    in:
      experiment:
        source: Export_PSL/experiment
        valueFrom: ${ return [ self ]; }
      name:
        default: "Regrid PSL"
      command:
        default: "/home/jovyan/work/vorticity/regrid.sh"
      args:
        default: "0:70 100:320 r880x280"
      input:
        default: "/home/jovyan/work/vorticity/output/@{source_file}"
      output:
        default: "/home/jovyan/work/vorticity/output/regridded_@{source_file}"
    out: [experiment]

  Import_U:
    run: tasks/importnc2.cwl
    in:
      experiment: Iterate_on_time/experiment
      name:
        default: "Import U"
      src_path:
        default: "/home/jovyan/work/vorticity/input/ua_@{source_file+4}"
      measure:
        default: "ua"
      imp_dim:
        default: "lat"
      container: container
      subset_dims:
        default: "plev|lat|lon"
      subset_filter:
        default: "85000|0:70|100:320"
      subset_type:
        default: "coord"
      nfrag: nthreads
      nthreads: nthreads
    out: [experiment]
  Put_radius_factor:
    run: tasks/intercube.cwl
    in:
      experiment1: Import_U/experiment
      experiment2: Evaluate_radius_factor/experiment
      name:
        default: "Put radius factor"
      operation:
        default: "div"
      cube2_is_array:
        default: "yes"
    out: [experiment]
  Evaluate_du_dlat:
    run: tasks/apply.cwl
    in:
      experiment: Put_radius_factor/experiment
      name:
        default: "Evaluate du/dlat"
      query:
        default: "oph_matheval(oph_gsl_spline(measure,dimension,dimension,1),'180*x/(6372507*pi)')"
      measure_type:
        default: "auto"
    out: [experiment]
  Rollup_U:
    run: tasks/rollup.cwl
    in:
      experiment: Evaluate_du_dlat/experiment
      name:
        default: "Rollup U"
    out: [experiment]
  Permute_U:
    run: tasks/permute.cwl
    in:
      experiment: Rollup_U/experiment
      name:
        default: "Permute U"
      dim_pos:
        default: "2,1"
    out: [experiment]

  Import_V:
    run: tasks/importnc2.cwl
    in:
      experiment: Iterate_on_time/experiment
      name:
        default: "Import V"
      src_path:
        default: "/home/jovyan/work/vorticity/input/va_@{source_file+4}"
      measure:
        default: "va"
      imp_dim:
        default: "lon"
      container: container
      subset_dims:
        default: "plev|lat|lon"
      subset_filter:
        default: "85000|0:70|100:320"
      subset_type:
        default: "coord"
      nfrag: nthreads
      nthreads: nthreads
    out: [experiment]
  Evaluate_dv_dlon:
    run: tasks/apply.cwl
    in:
      experiment: Import_V/experiment
      name:
        default: "Evaluate dv/dlon"
      query:
        default: "oph_matheval(oph_gsl_spline(measure,dimension,dimension,1),'180*x/(6372507*pi)')"
      measure_type:
        default: "auto"
    out: [experiment]
  Rollup_V:
    run: tasks/rollup.cwl
    in:
      experiment: Evaluate_dv_dlon/experiment
      name:
        default: "Rollup V"
    out: [experiment]
  Put_latitude_factor:
    run: tasks/intercube.cwl
    in:
      experiment1: Rollup_V/experiment
      experiment2: Evaluate_latitude_factor/experiment
      name:
        default: "Put latitude factor"
      operation:
        default: "div"
      cube2_is_array:
        default: "yes"
      extension_type:
        default: "interlace"
    out: [experiment]

  Evaluate_vorticity:
    run: tasks/intercube.cwl
    in:
      experiment1: Permute_U/experiment
      experiment2: Put_latitude_factor/experiment
      name:
        default: "Evaluate vorticity"
      measure: output_variable2
    out: [experiment]

  Delete_metadata:
    run: tasks/metadata.cwl
    in:
      experiment: Evaluate_vorticity/experiment
      name:
        default: "Delete metadata"
      mode:
        default: "delete"
      variable: output_variable2
      metadata_key:
        default: "comment"
    out: [experiment]
  Update_metadata:
    run: tasks/metadata.cwl
    in:
      experiment: Delete_metadata/experiment
      name:
        default: "Update metadata"
      mode:
        default: "update"
      variable: output_variable2
      metadata_key:
        default: "standard_name|long_name|units"
      metadata_value:
        default: "atmosphere_relative_vorticity|Vorticity (relative)|s**-1"
      force:
        default: "yes"
    out: [experiment]
  Export_VO:
    run: tasks/exportnc2.cwl
    in:
      experiment: Update_metadata/experiment
      name:
        default: "Export VO"
      output:
        default: "/home/jovyan/work/vorticity/output/vo_@{source_file+4}"
    out: [experiment]
  Regrid_VO:
    run: tasks/generic.cwl
    in:
      experiment:
        source: Export_VO/experiment
        valueFrom: ${ return [ self ]; }
      name:
        default: "Regrid VO"
      command:
        default: "/home/jovyan/work/vorticity/regrid.sh"
      args:
        default: "0:70 100:320 r880x280"
      input:
        default: "/home/jovyan/work/vorticity/output/vo_@{source_file+4}"
      output:
        default: "/home/jovyan/work/vorticity/output/regridded_vo_@{source_file+4}"
    out: [experiment]

  Merge_outputs:
    run: tasks/generic.cwl
    in:
      experiment: [Regrid_PSL/experiment, Regrid_VO/experiment]
      name:
        default: "Merge outputs"
      command:
        default: "/home/jovyan/work/vorticity/merge.sh"
      args: output_variable1
      input:
        default: "/home/jovyan/work/vorticity/output/regridded_@{source_file}|/home/jovyan/work/vorticity/output/regridded_vo_@{source_file+4}"
      output:
        default: "/home/jovyan/work/vorticity/output/both_@{source_file+4}"
    out: [experiment]

  End_iteration:
    run: tasks/endfor.cwl
    in:
      experiment:
        source: Merge_outputs/experiment
        valueFrom: ${ return [ self ]; }
      name:
        default: "End iteration"
    out: [experiment]

  Destroy_the_work_container:
    run: tasks/deletecontainer.cwl
    in:
      experiment: End_iteration/experiment
      name:
        default: "Destroy the work container"
      container: container
      force:
        default: "yes"
      on_error:
        default: "skip"
    out: [experiment]

