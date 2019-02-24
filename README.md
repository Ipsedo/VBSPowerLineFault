# VBSPowerLineFault
_Samuel Berrien_, _Thierry Cabanal_

Challenge Kaggle [VBS Power Line Fault Detection](https://www.kaggle.com/c/vsb-power-line-fault-detection)

## Usage

__Attention__ : Bien s'assurer que les données sont dans `/path/to/VBSPowerLineFault/data` de la manière suivante :
```
VBSPowerLineFault
    |-- data
    |    |-- metadata_test.csv
    |    |-- metadata_train.csv
    |    |-- sample_submission.csv
    |    |-- test.parquet
    |    |-- train.parquet
    |
    |-- ...    

```
Pour lancer le projet :

```bash
$ cd VBSPowerLineFault
$ python separate_data.py
$ python main.py
$ python test.py
```