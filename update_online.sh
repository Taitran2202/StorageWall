#!/usr/bin/bash

# variables
PATH="/home/greystone/ML_TEMPLATE/ML_MODULE"
MYENV="env_main_app"
MYCONDA="miniconda3_py38"
REQUIREMENTS="${PATH}/app/requirements.txt"
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

# install requirements
echo "Updating requirements..."

$PIP install -r $REQUIREMENTS

echo "Done update!"

exit 0 # notify success