# coco_investigator

This is a script to extract useful informatio from a coco dataset (JSON) file.
## What is does:
* loading coco Dataset using coc api
* spliting image sets (train, valid, eval) using image IDs: 1. random, 2. folder based, 3.patch base
* computing balacing weights for both sub categories and super categories
* computing areas, instance nums ,... for both sub and super categories in the dataset