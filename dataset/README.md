## Setup DVC
- Config server
```
dvc remote add -d master ssh://mldataset@mlserver.greystonevn.com:2222/home/mldataset/<project_dataset_folder>/
```

- Modify server (if you want to update project dataset folder)
```
dvc remote modify master url ssh://mldataset@mlserver.greystonevn.com:2222/home/mldataset/<new_project_dataset_folder>/
```

- Config credential info
```
dvc remote modify master user mldataset
dvc remote modify master password mldataset
```

- Naming convention
  + Please set name for ML model folder as follows: "model_\<func_name\>_\<technique\>"
  + Example: "model_detect_MBA_SVTR", "model_detect_phone_on_tray_MASKRCNN", "model_ocr_label_back_side_SVTR"
