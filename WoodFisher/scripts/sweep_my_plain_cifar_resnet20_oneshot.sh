##### instructions ####
# first generate commands via `bash filename> > commands.txt` and then execute `bash commands.txt`
#######################

#!/bin/bash

TARGET_SPARSITYS=(0.8) # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.999
MODULES=("conv1_segments.0.0.conv1_segments.0.0.conv2_segments.0.1.conv1_segments.0.1.conv2_segments.0.2.conv1_segments.0.2.conv2_segments.1.0.conv1_segments.1.0.conv2_segments.1.1.conv1_segments.1.1.conv2_segments.1.2.conv1_segments.1.2.conv2_segments.2.0.conv1_segments.2.0.conv2_segments.2.1.conv1_segments.2.1.conv2_segments.2.2.conv1_segments.2.2.conv2_fc")

SEEDS=(0) # 0 1 2
FISHER_SUBSAMPLE_SIZES=(1000)
PRUNERS=(globalmagni) # globalmagni magni woodfisherblock diagfisher
JOINTS=(1)
FISHER_DAMP="1e-5"
EPOCH_END="1"
PROPER_FLAG="1"
ROOT_DIR=".."
DATA_DIR="../../../Linear_Mode_Connectivity/data"
SWEEP_NAME="my_plain_cifar_resnet20_oneshot"
NOWDATE=""
DQT='"'
GPUS=(0)
LOG_DIR="${ROOT_DIR}/${SWEEP_NAME}/log/"
CSV_DIR="${ROOT_DIR}/${SWEEP_NAME}/csv/"
mkdir -p ${LOG_DIR}
mkdir -p ${CSV_DIR}

ID=0

for PRUNER in "${PRUNERS[@]}"
do
    for JOINT in "${JOINTS[@]}"
    do
        if [ "${JOINT}" = "0" ]; then
            JOINT_FLAG=""
        elif [ "${JOINT}" = "1" ]; then
            JOINT_FLAG="--woodburry-joint-sparsify"
        fi

        for SEED in "${SEEDS[@]}"
        do
            for MODULE in "${MODULES[@]}"
            do
                for TARGET_SPARSITY in "${TARGET_SPARSITYS[@]}"
                do
                    for FISHER_SUBSAMPLE_SIZE in "${FISHER_SUBSAMPLE_SIZES[@]}"
                    do
                        if [ "${FISHER_SUBSAMPLE_SIZE}" = 80 ]; then
                            FISHER_MINIBSZS=(1)
                        elif [ "${FISHER_SUBSAMPLE_SIZE}" = 1000 ]; then
                            FISHER_MINIBSZS=(1)
                        elif [ "${FISHER_SUBSAMPLE_SIZE}" = 5000 ]; then
                            FISHER_MINIBSZS=(1)
                        fi

                        for FISHER_MINIBSZ in "${FISHER_MINIBSZS[@]}"
                        do


                            echo CUDA_VISIBLE_DEVICES=${GPUS[$((${ID} % ${#GPUS[@]}))]} python ${ROOT_DIR}/main.py --exp_name=${SWEEP_NAME} --dset=cifar10 --dset_path=${DATA_DIR} --arch=my_plain_cifar_resnet20 --config_path=${ROOT_DIR}/configs/my_resnet20_woodfisher.yaml --workers=1 --batch_size=64 --logging_level debug --gpus=0 --from_checkpoint_path ${ROOT_DIR}/checkpoints/cifar10/my_plain_cifar_resnet20.pt --batched-test --not-oldfashioned --disable-log-soft --use-model-config --sweep-id ${ID} --fisher-damp ${FISHER_DAMP} --prune-modules ${MODULE} --fisher-subsample-size ${FISHER_SUBSAMPLE_SIZE} --fisher-mini-bsz ${FISHER_MINIBSZ} --update-config --prune-class ${PRUNER} --target-sparsity ${TARGET_SPARSITY} --prune-end ${EPOCH_END} --prune-freq ${EPOCH_END} --result-file ${CSV_DIR}/prune_module-woodfisher-based_all_epoch_end-${EPOCH_END}.csv --seed ${SEED} --deterministic --full-subsample ${JOINT_FLAG} --fisher-optimized --fisher-parts 5 --offload-grads --offload-inv '&>' ${LOG_DIR}/${PRUNER}_proper-1_joint-${JOINT}_module-all_target_sp-${TARGET_SPARSITY}_epoch_end-${EPOCH_END}_samples-${FISHER_SUBSAMPLE_SIZE}_${FISHER_MINIBSZ}_damp-${FISHER_DAMP}_seed-${SEED}.txt

                            ID=$((ID+1))

                        done
                    done
                done
            done
        done
    done
done