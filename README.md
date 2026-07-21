# Actimeter Annotation

## Background 
The repository contains code for a project conducted at the University of Geneva in the [Sleep and Cognition Neuroimaging Laboratory](https://neurocenter-unige.ch/research-groups/sophie-schwartz/), supervised by Virginie Sterpenich. The aim is to develop a semi-automatic pipeline to annotate actigraph data from [ActiLife v6.13.6](https://www.actigraphcorp.com/release-notes/?prod=ActiLife6) in order to derive the main sleep metrics. 
The pipeline is using [PyActigraph](https://ghammad.github.io/pyActigraphy/), and allow visualisation as well as manual validation of the algorithm derived sleep/wake windows. 

## Files structure
```
├── run_pipeline.py
├── plots.py
├── config.py
├── config.yaml
```

## Implementation 

`run_pipeline.py` :

`plots.py` : 

`config.py` : 

`config.yaml` : 

## Usage 
After having install the proprer environment, you can run the following command: 

```bash
python run_pipeline.py --agd    "*.agd" --diary  ".csv"
```

If you not willing to use the pipelie iwht a diary you can simply use : 

```bash
python run_pipeline.py --agd    "*.agd"
```

## Compatibility  
The codes have been tested on versions of Python 3.9.23 
