#!usr/bin/bash

# check setup folder exist, if yes remove setup file before copy new
DIR=$PWD/appPython
echo "Directory to build: $DIR"
if [ ! -d $DIR ]; then
    mkdir $DIR
    echo "Created: $DIR"
else
    echo "Existed! $DIR"
    echo "Please remove: $DIR"
    exit -1
fi

# remove pycache
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf
#find . | grep -E "(__pycache__|\.pyc$)" | xargs rm -rf

# copy files
cp globals.py $DIR
cp main_app.py $DIR
cp ml_subapp_environment.yml $DIR
cp pre_install.sh $DIR
cp README.md $DIR
cp requirements.txt $DIR
cp signal_handler.py $DIR
cp update_offline.sh $DIR
cp update_online.sh $DIR
cp config.json $DIR
cp calib.csv $DIR

# copy folers
cp -r assets $DIR
cp -r controllers $DIR
cp -r modules $DIR
cp -r network $DIR
cp -r utils $DIR
cp -r images $DIR

echo "Completed!"
exit 0
