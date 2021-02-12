
import random
import yaml
import os
import argparse
import numpy as np
from pycocotools.coco import COCO


class CocoInvestigator():
    def __init__(self, _root, _dataset_name):
        # load in the files that exist in the different paths.
        self.root_dir = _root
        self.datsset_name = _dataset_name
        # build up coco dataset Dir
        self.dataset_dir = self.root_dir + self.datsset_name + "/" + self.datsset_name + ".json"
        # load Coco dataset
        self.coco = COCO(self.dataset_dir)

        #  getting all img id in the dataset
        self.totalList = [d['id'] for (index, d) in enumerate(self.coco.dataset['images'])]
        # print("total image IDs:", len(self.totalList))

        self.annotated_area = 0

        #  getting category ids
        self.catIDs = self.coco.getCatIds()
        self.catLabels = self.getCatLabels()

        self.img_width = self.coco.dataset['images'][0]['width']
        self.img_height = self.coco.dataset['images'][0]['height']

        self.yamlName = self.root_dir + '/' + self.datsset_name + '/' + self.datsset_name + '.yaml'


        self.total_area = 0
        self.annotated_area = 0

        # paramters refering to only Train set !
        self.datasetMetaData = {"total_area": 0,
                                "annotated_area": 0,
                                "cat_labels": [],
                                "areas":[],
                                "weights":[],
                                "instance_num":[]}
        self.imgSets = {"train": [], "valid": [], "eval": []}

    def getImageSetsRandomly(self):
        #  TODO check randomness !!!!
        random.seed(30)
        # deviding imgae list and subctracting from original list
        self.imgSets["train"] = random.sample(self.totalList, len(self.totalList)//2)
        remaining = list(set(self.totalList)^set(self.imgSets["train"]))
        self.imgSets["valid"] = random.sample(remaining, len(self.totalList)//4)
        remaining = list(set(self.totalList)^set(self.imgSets["train"])^set(self.imgSets["valid"]))
        self.imgSets["eval"] =  random.sample(remaining, len(self.totalList)//4)

        print("imgSets.train Len:", len(self.imgSets["train"]),
              "imgSets.valid Len:", len(self.imgSets["valid"]),
              "imgSets.eval Len:", len(self.imgSets["eval"]))
    # TODO check validity of imagse sets used for gettgin stats
    def getCatImageIDs(self, imageSet):
        # finding the ids of images in image set
        imageSetIDs = []
        for image_path in imageSet:
            imageSetIDs.append((next((d['id'] for (index, d) in \
                                enumerate(self.coco.dataset['images']) if d["id"] == image_path), None)))
        return imageSetIDs

    def getCatsAreas(self, catList, imageSet):
        #  geting area of all categories individually
        cats_areas = []
        cats_instance = []
        for cat_id in catList:
            cat_area = 0
            cat_instances = 0

            for (index, d) in  enumerate(self.coco.dataset['annotations']):
                if d["image_id"] in imageSet and d["category_id"] == cat_id :
                    cat_area += d["area"]
                    cat_instances += 1

            if cat_area == None:
                cat_area = 0

            cats_areas.append(cat_area)
            cats_instance.append(cat_instances)

        return cats_areas, cats_instance

    def getCatLabels(self)  :
        # display COCO categories and supercategories
        self.cats = self.coco.loadCats(self.catIDs)
        self.cat_labels = [cat['name'] for cat in self.cats]
        return self.cat_labels

    def getDirImgs(self, dir):
        # finding the ids of images in image set
        currPatchImgs = []
        for img in self.coco.dataset["images"]:
            dirList = img["path"].split("/")
            for imgPath in dirList:
                if imgPath == dir:
                    currPatchImgs.append(img["id"])
                    # print(currPatchImgs)

        return currPatchImgs

    def getPatchNum(self):
        self.patchList = list()
        for root, dirs, files in os.walk("/home/alireza/data/Dataset/CKA_corn_2020/", topdown=True):
            for name in dirs:
                if not name[0] == '.' and name[0:5] == "patch" :
                    self.patchList.append(os.path.join(root, name))

        # print(self.patchList, len(self.patchList))
        return  len(self.patchList)

    def computeCatsBalancingWeights(self, areasList, catList, epsillon=1.02):
        weights = []
        # Calculating weights for each category
        for cat_id in range(len(catList)):
            # weight score of each category w.r.t bg
            fc = areasList[cat_id] / areasList[0] #bg
            # computing weight of category
            cat_weight = 1/np.log(fc + epsillon)
            # adding computed weight to weight list
            weights.append(float(cat_weight))

        weights = list(weights)

        return weights

    def getImageSetsFromYaml(self, yamlName):
        with open(self.yamlName, 'r') as file:
            imgSetyYaml = yaml.load(file,  Loader=yaml.FullLoader)

        return imgSetyYaml["image_sets"]

    def updateYamlFile(self , metaDataDict):
        with open(self.yamlName, 'w') as file:
            documents = yaml.dump(metaDataDict, 
                                  file,
                                  encoding='utf-8', 
                                  allow_unicode=True, 
                                  default_flow_style=False)

    def getImageNames(self):
        namelist = list()
        for img in self.coco.dataset["images"]:
            dirList = img["path"].split("/")
            namelist.append(dirList[-1].replace('.png', ''))

        return namelist

    def makeImgSetsFromDir(self, dirList, imgSet="train"):
        for _dir in dirList:
            cocoInv.imgSets[imgSet].extend(cocoInv.getDirImgs(_dir))

    def getSuperClassInstanNum(self, imgSet):
        self.superCats = list()
        # getting a list of super Cats in dataset
        for sCat in self.coco.dataset["categories"]:
            self.superCats.append(sCat["supercategory"])
        # adding bg as a super Cat
        self.superCats.append("bg")
        # finding unique super Cats in the dataset
        self.superCats = list(np.unique(self.superCats))
        # get super Cats of each nanotation and thier areas 
        self.superCatList = list()
        for ann in self.coco.dataset["annotations"]:
            for cat in self.coco.dataset["categories"]:
                if ann["category_id"] == cat["id"]:
                    annSCat =cat["supercategory"]
                    self.superCatList.append(annSCat)
                    
        #  make a dictionary of super Cats names and the instace numbers
        self.superCatsInstanceNum = dict()
        for sCat in self.superCats:
            catIntsnceNum = self.superCatList.count(sCat)
            self.superCatsInstanceNum.update({sCat:catIntsnceNum})
        
        print(self.superCatsInstanceNum)
        
    def getSuperClassInstanAreas(self, imgSet, imgSetArea):
        self.superCats = list()
        # getting a list of super Cats in dataset
        for sCat in self.coco.dataset["categories"]:
            self.superCats.append(sCat["supercategory"])
        # adding bg as a super Cat
        self.superCats.append("bg")
        # finding unique super Cats in the dataset
        self.superCats = list(np.unique(self.superCats))
        # get super Cats of each nanotation and thier areas 
        self.superCatList = list()
        
        
        self.superCatsArea = dict()
        for sCat in self.superCats:
            self.superCatsArea.update({sCat:0})
            
        for ann in self.coco.dataset["annotations"]:
            annArea = ann["area"] 
            for cat in self.coco.dataset["categories"]:
                if ann["category_id"] == cat["id"]:
                    self.superCatsArea[cat["supercategory"]] += annArea
                    
        
        totalTrainSetArea = self.img_height * self.img_width * len(imgSet)
        totalAnnotatedArea = imgSetArea
        backGroundArea = totalTrainSetArea - totalAnnotatedArea
        print(backGroundArea , totalTrainSetArea , totalAnnotatedArea)
        self.superCatsArea["bg"] = backGroundArea
        
        print(self.superCatsArea)  
        
    def getsuperCatsWeights(self):
        areaList = list(self.superCatsArea.values())
        catList = list(self.superCatsInstanceNum.keys())
        print(areaList, catList)
        self.superCatsWeights = self.computeCatsBalancingWeights(areaList, catList)
        print(self.superCatsWeights)        
    


if __name__ == '__main__':
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("root_dir", help="root directory of dataset")
    parser.add_argument("dataset_name", help="dataset name !")
    parser.add_argument("--splitMode",  type=str, default="random", required=True)
    parser.add_argument("--train", nargs="+", help="Dir of folders belonging to train set", required=False)
    parser.add_argument("--valid", nargs="+", help="Dir of folders belonging to valid set", required=False)
    parser.add_argument("--eval", nargs="+",  help="Dir of folders belonging to eval set", required=False)

    args = parser.parse_args()
    cocoInv = CocoInvestigator(args.root_dir, args.dataset_name)
    trainSetCatIDs = cocoInv.catIDs

    if args.splitMode == "random":
        cocoInv.getImageSetsRandomly()

    elif args.splitMode == "dir_base":
        imgList = list()
        for set in ["train", "valid", "eval"]:
            if set =="train":
                imgList = args.train
            elif set == "valid":
                imgList = args.valid
            elif set == "eval":
                imgList = args.eval
            cocoInv.makeImgSetsFromDir(imgList, imgSet=set)

    elif args.splitMode == "patch_base":
        print(cocoInv.datasetMetaData)
        print(trainSetWeights)
        number_of_patches = 15
        for i in range(1, number_of_patches):
            imgsInDir = cocoInv.getDirImgs("patch_"+str(i))
            areas, instnces = cocoInv.getCatsAreas(cocoInv.catIDs, imgsInDir)

            totalArea = cocoInv.img_height * cocoInv.img_width * len(imgsInDir)
            totalAnnotatedArea = sum(areas)
            backGroundArea = totalArea - totalAnnotatedArea
            areas[idx] = backGroundArea

            # weights = cocoInv.computeCatsBalancingWeights(areas, imgsInDir)
            print("patch"+ str(i), instnces)

    elif args.splitMode == "yaml_base":
        cocoInv.imgSets = cocoInv.getImageSetsFromYaml("args.dataset_name"+".yaml")

    # print(cocoInv.imgSets["train"])

    
    trainSetAreas, trainSetInstnces = cocoInv.getCatsAreas(trainSetCatIDs, cocoInv.imgSets["train"])

    totalTrainSetArea = cocoInv.img_height * cocoInv.img_width * len(cocoInv.imgSets["train"])
    totalAnnotatedArea = sum(trainSetAreas)

    backGroundArea = totalTrainSetArea - totalAnnotatedArea
    idx = 0  # bg idx in list
    trainSetCatIDs.insert(idx, 0)
    trainSetAreas.insert(idx, backGroundArea)
    trainSetInstnces.insert(idx, 0)
    cocoInv.catLabels.insert(idx, "bg")
    # print(cocoInv.catLabels)

    cocoInv.getSuperClassInstanNum(cocoInv.imgSets["train"])
    cocoInv.getSuperClassInstanAreas(cocoInv.imgSets["train"], totalAnnotatedArea)
    cocoInv.getsuperCatsWeights()

    trainSetWeights = cocoInv.computeCatsBalancingWeights(trainSetAreas, trainSetCatIDs)

    datasetMetaData =  {"image_size":{"width":cocoInv.img_width,
                                      "height":cocoInv.img_height},
                        "class_num":len(cocoInv.catLabels),
                        "class_labels":cocoInv.catLabels,
                        "class_ids":trainSetCatIDs,
                        "class_instances":trainSetInstnces,
                        "class_areas":trainSetAreas,
                        "class_weights":trainSetWeights,
                        "image_sets":{"train":cocoInv.imgSets["train"],
                                      "valid":cocoInv.imgSets["valid"],
                                      "eval": cocoInv.imgSets["eval"]},
                        "super_categories":{"areas": list(cocoInv.superCatsArea.values()),
                                          "instace_nums": list(cocoInv.superCatsInstanceNum.values()),
                                          "weights": cocoInv.superCatsWeights,
                                          "names": [str(x) for x in cocoInv.superCats]}}

    cocoInv.updateYamlFile(datasetMetaData)

    for set in ["train", "valid", "eval"]:
        areas, instnces = cocoInv.getCatsAreas(cocoInv.catIDs, cocoInv.imgSets[set])
        print(set, len(cocoInv.imgSets[set]), instnces)


# python cocoDatasetMan.py /home/alireza/nfs/cube3/data1/alireza/datasets/ CKA_corn_2020 --splitMode dir_base  --train patch_1 --valid patch_2 --eval patch_3

