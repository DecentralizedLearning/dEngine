if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Not in a virtualenv"
    exit 1
fi

if [[ -z "${SEED}" ]]; then
    echo "{SEED} env variable missing"
    exit 1
fi

: "${OUTPUT_DIRECTORY_PREFIX:=logs}"
OUTPUT_DIRECTORY_PREFIX="${OUTPUT_DIRECTORY_PREFIX%/}"  # Remove trailing slash if needed

# ............... #
# CONFIGURATION
# ............... #
DEFAULT_OPTIONS="--verbosity debug --seed ${SEED}"

if [[ -n "${SANITY_CHECK}" ]]; then
  DEFAULT_OPTIONS+=" --sanity_check"
fi

DEFAULT_CONFIG_OVERRIDES=" \
    client.training_engine.arguments.training_batch_size=1024 \
    partitioning.arguments.validation_percentage=0
    dataset_train.arguments.augmentations=False \
    dataset_test.arguments.augmentations=False
"

simulate \
    --config core/datasets/mnist.yml \
    core/scenarios/centralized.yml \
    core/partitioning/iid.yml \
    configs/memnet_1M.yml \
    ${DEFAULT_OPTIONS} \
    --output_directory ${OUTPUT_DIRECTORY_PREFIX}/GAN name=mnist,GAN \
    client.training_engine.arguments.epochs=250 \
    $DEFAULT_CONFIG_OVERRIDES

simulate \
    --config core/datasets/cifar10.yml \
    core/scenarios/centralized.yml \
    core/partitioning/iid.yml \
    configs/memnet_30M.yml \
    ${DEFAULT_OPTIONS} \
    --output_directory ${OUTPUT_DIRECTORY_PREFIX}/GAN name=cifar10,GAN \
    client.training_engine.arguments.epochs=500 \
    $DEFAULT_CONFIG_OVERRIDES

simulate \
    --config core/datasets/cifar100.yml \
    core/scenarios/centralized.yml \
    core/partitioning/iid.yml \
    configs/memnet_30M.yml \
    ${DEFAULT_OPTIONS} \
    --output_directory ${OUTPUT_DIRECTORY_PREFIX}/GAN name=cifa100,GAN \
    client.training_engine.arguments.epochs=500 \
    $DEFAULT_CONFIG_OVERRIDES
