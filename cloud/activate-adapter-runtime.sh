#!/usr/bin/env bash
# Isolated optional LoRA dependencies; do not change an in-flight student/teacher environment.
source /root/rivermind-data/FIRE_R2/cloud/activate-teacher-runtime.sh || return 1
export PYTHONPATH=/FIRE_R2-runtime/adapter-site-packages:$PYTHONPATH
