"""
Create a class that organizes all of the COCO data. It might store the following:

-All the image IDs
-All the caption IDs
-Various mappings between image/caption IDs, and associating caption-IDs with captions
image-ID -> [cap-ID-1, cap-ID-2, ...]
caption-ID -> image-ID
caption-ID -> caption (e.g. 24 -> "two dogs on the grass")

"""
from cogworks_data.language import get_data_path

from pathlib import Path
import json

class CocoData():
    def __init__(self, path):
        self.filename = get_data_path(path)
        with Path(self.filename).open() as f:
            self.coco_data = json.load(f)
        self.urls = {}
        for u in range(len(self.coco_data["images"])):
            self.urls[self.coco_data["images"][u]["id"]] = self.coco_data["images"][u]["coco_url"]
        self.image_ids = {}
        for x in range(len(self.coco_data["images"])):
            self.image_ids[(self.coco_data["images"][x]["id"])] = []
        for y in range(len(self.coco_data["annotations"])):
            self.image_ids[self.coco_data["annotations"][y]["image_id"]].append(self.coco_data["annotations"][y]["id"])
        self.captions = {}
        for z in range(len(self.coco_data["annotations"])):
            self.captions[self.coco_data["annotations"][z]["id"]] = self.coco_data["annotations"][z]["caption"]        
    
    def get_image_ids(self):
        return list(self.image_ids.keys())
    
    def get_caption_ids(self):
        return [cap_id for cap_ids in self.image_ids.values() for cap_id in cap_ids]

    def get_captions(self):
        all_captions = []

        for cap_ids in self.image_ids.values():
            for cap_id in cap_ids:
                if cap_id in self.captions:
                    all_captions.append(self.captions[cap_id])
        return all_captions
                    

    def get_image_id_from_caption_id(self,cap_id):
        for index, (image_id, cap_ids) in enumerate(self.image_ids.items()):
            if cap_id in cap_ids:
                return image_id
        return "error: cap_id does not exist"
    
    def get_caption_from_caption_id(self,cap_id):
        if cap_id in self.captions:
            return self.captions[cap_id]
        else:
            return "error: cap_id does not exist"
    
    def get_captions_from_image_id(self, image_id):
        if image_id in self.image_ids:
            captions = []
            for cap_id in self.image_ids[image_id]:
                captions.append(self.get_caption_from_caption_id(cap_id))
            return captions
        else:
            return "error: image_id does not exist"
    
    def get_url_from_image_id(self, image_id):
        if image_id in self.urls:
            return self.urls[image_id]
        else:
            return "error: image_id does not exist"


#Testing: 

data = CocoData("captions_train2014.json")
print(data.get_captions_from_image_id(318556))


