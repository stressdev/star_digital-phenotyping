# STAR Digital Phenotyping Pipeline

The file `tasks.py` organizes the tasks that can be completed by these scripts using the invoke library. The file `invoke.yaml` specifies some variable values used by `tasks.py`.

In order to run invoke, you need to first load the python module, and then load the `star_diph` conda environment (which must be created for your user using the `star_diph.yml` file.

## Example environment creation

```{bash}
module load python/3.10.9-fasrc01 
mamba env create -f star_diph.yml
```

## Example environment activation and `invoke`

The command, `invoke -l` below will list the tasks. To get help for a task, e.g., the build task, you can run `invoke -h build`.

```{bash}
module load python/3.10.9-fasrc01
mamba activate star_diph
invoke -l
```

## Other necessary info

In order to work properly, these scripts need an updated csv taken from the [STAR Participant Tracker Google Sheet](https://docs.google.com/spreadsheets/d/1G_4CMYGZfR_GInQeMBgVMAx5ZsTYRjqk/edit?usp=sharing&ouid=104706038744730652520&rtpof=true&sd=true) which should be saved in the same directory as this README file and named `STAR_Month-Day_Subject-Calculator.xlsx - Sheet1.csv`. 

In order to make sure that the script is using the most up-to-date phone and call logs, please run

```{bash}
invoke sync-phone-data
```

To build all data and reports, run

```{bash}
invoke build -a
```

This will submit multiple sbatch jobs.
