# BSP-AF-S5

## Set up
1) Create a python environment
2) run `pip install -r requirements.txt`

## Mapshaper set up
1) run `conda install -c conda-forge nodejs`
2) run `npm install -g mapshaper`
3) cd to `python/geodata`
4) run 
```bash
mapshaper \
  -i countries.json \
  -proj eqearth \
  -simplify visvalingam keep-shapes 10% \
  -proj wgs84 \
  -o out_countries.json format=geojson precision=0.0001 force
```
Ps. you can run even 1%
5) run python/geodata/simplify_geodata.py 

## Notes
The `src/geodata/viz` is defunct and to be removed later on.


