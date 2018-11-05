import json

#### DETECT TOOLS
def load_annot_json(filename):

    ######### ANNOT FILE
    print("Loading JSON annotations file",filename)
    with open(filename) as f:
         data = json.load(f)
    f.close()

    categories=data['categories']
    catlen=len(categories)
    print ("Categories in annotation file:",catlen)

    images=data['images']
    imglen=len(images)
    print ("Images in annotation file:",imglen)

    boxes=data['annotations']
    boxlen=len(boxes)
    print ("Boxes in annotation file:",boxlen)

    return categories,catlen,images,imglen,boxes,boxlen
