#!/bin/bash

npoint=30 # number of sites on one side of the lattice (that will extend for npoint x npoint sites)
nmeas=1000 # number of energy measures
NHMC=12 # number of integration steps for the Monte-Carlo trajectory
dt=0.05 # size of the stochastic time-step`
ncomp=14 # number of independent components. The theory simulated will be O(N) with N = ncomp+1
ptord=10 # maximum perturbative order
every=7 # number of measures to skip after each measurement

source configuration.conf

name="O($((ncomp+1)))_Npoint${npoint}_dt${dt}_ord${ptord}_NHMC${NHMC}_nMeas${nmeas}_every${every}"
resume_fname="../checkpoint/OMF_HMC_${name}.jld"
lat_fname="${SITE_ENERGY_PATH}/${name}.mat"

SBATCH_OPTS=(
    --job-name=$name
    --partition=gpu
    --qos=gpu
    --gres=gpu:a100_80g:1
    --mem=40G
    --time=1-0:00:00
    --output=logs/%x_%j.out
    --error=logs/%x_%j.err
    --mail-user=$NOTIFICATION_EMAIL
    --mail-type=ALL
)

# sbatch ${SBATCH_OPTS[@]} job_sbatch.bash $npoint $nmeas $NHMC $dt $ncomp $ptord $every # to start
sbatch ${SBATCH_OPTS[@]} job_sbatch.bash $npoint $nmeas $NHMC $dt $ncomp $ptord $every $lat_fname # to start, with site-per-site energy to be saved
# sbatch ${SBATCH_OPTS[@]} job_sbatch.bash $resume_fname # to resume
