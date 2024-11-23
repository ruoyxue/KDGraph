# Prepare SpaceNet3 for KDGraph

1. Convert images in **PS-RGB directory** to 8 bit-RGB.
```Python
python convert_to_8bit.py
```

2. Generate cropped training dataset for KDGraph based on **geojson_roads** in SpaceNet3.
   
```Python
python generate_cropped_dataset.py
```

3. Generate pickle graphs for APLS evaluation.

```Python
python spacenet_transform.py
python convert_to_dict.py
```
