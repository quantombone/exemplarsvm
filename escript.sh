#!/bin/sh
matlab -r "addpath(genpath(pwd)); esvm_script_train_dt('"$1"');"