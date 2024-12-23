#!/usr/bin/bash

# variables
PATH="/home/greystone/ML_TEMPLATE/ML_MODULE"
MYENV="env_main_app"
MYCONDA="miniconda3_py38"
PACKAGES="${PATH}/packages"
CONDA="${PATH}/${MYCONDA}/bin/conda"
ENV="${PATH}/${MYCONDA}/envs/${MYENV}"
PIP="${PATH}/${MYCONDA}/envs/${MYENV}/bin/pip"

# check miniconda3 installation
if [ -f "$CONDA" ]; then
    echo "Miniconda3 is installed"
else
    echo "Miniconda3 is NOT installed"
    #exit bash script
    exit 1 # notify failue
fi

# check conda virtualenv installation
if [ -d "$ENV" ]; then
    echo "${MYENV} is created"
else
    echo "${MYENV} is NOT created"
    #exit bash script
    exit 1 # notify failue
fi

# loop over wheel files and install
for item in $PACKAGES/*.whl
do
    echo "Installing: $item"
    $PIP install $item
done

# loop over tar.gz and install
for item in $PACKAGES/*.tar.gz
do
    echo "Installing: $item"
    $PIP install $item
done

echo "Done update!"

exit 0 # notify success