# Baseline test for Adaptdl

To submit jobs, run `run_workload.py`. Note that `cache_images(templates)` shold not be called repeatedly on the same cluster. `workloads-6.csv` documents the submitting sequence.

`run_monitor.py`monitors all the adaptdl jobs, which generates a json file documenting all live jobs, a seperate json file for each job. This should be executes when running workloads.

`save_pod_log.py` documents the logs of all pods in the cluster by creating a txt for each pod. Run this during submiting job since some pods may be terminated for other reasons and logs on them cannot be retrieved. This should be executes when running workloads.

`delete_adaptdljob.py` conveniently delete all the adaptdl jobs.

After submitting jobs, some necessary logs are stored in the pvc named `pollux`, which contains two directories: `/checkpoint` and `/tensorboard`. Each job will create a unique directory inside them. `train.txt`, `valid.txt`, and `overall_results.txt` can be retrieved in the subpaths of `/checkponit`, which record measures including training time, num samples, losses, accuracies... Note that the two directories should be emptied when a round of test begins.