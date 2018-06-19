Preparation of data for training

**Labeling tool**
https://github.com/tzutalin/labelimg

**Generate tf record**
python3 generate_record.py --csv_input=data/train_labels.csv --output_path=data/train.record