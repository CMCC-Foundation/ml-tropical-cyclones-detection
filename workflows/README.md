<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

# Workflows for Tropical Cyclones Detection

<p align="justify"> This repository includes a workflow useful to test tropical cyclone detection. </p>

## Ophidia workflow

<p align="justify"> Ophidia (https://ophidia.cmcc.it) is an open-source HPC framework for data-intensive analysis, exploiting advanced parallel computing techniques and smart data distribution methods. Ophidia exploits a client-server approach; the user can interact by means a Python module, called PyOphidia (https://pyophidia.readthedocs.io/en/latest/), using a JSON interface. </p>
<p align="justify"> A sample workflow is coded in "vorticity.py" and a Python notebook 'vorticity.ipynb' is also provided. It allows to evaluate the relative vorticity to be passed to ML algorithm for cyclone detection, thus performing a pre-processing phase for the inference. </p>
<p align="justify"> Assuming that input datasets are the Sea Level Pressure (psl) and the components of wind speeds (ua and va), the workflow: </p>

<ul>
<li>selects the variables from input NetCDF files related to a given space domain (North Pacific),</li>
<li>combines the variables to produce the vorticity using the formula: $dv/dx - du/dy$</li>
<li>regrids psl and vorticity to a common grid,</li>
<li>save psl and vorticity in the same output file.</li>
</ul>

<p align="justify"> The workflow can be started by the following Python command. </p>

```
$ vorticity()
```
<p align="justify"> CWL and JSON implemetations are also provided. Provenance information can be also produced by PyOphidia, once the execution is completed. </p>

