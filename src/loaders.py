from PIL import Image, ImageOps
import numpy as np

def load_list_file_class_to_numpy(filename,R,C,CH,RES):
    file = open(filename)
    c=0
    for line in file:
        fields = line.strip().split()
        c=c+1
    file.close()

    file = open(filename)

    X=np.zeros((c,R,C,CH))
    L=np.zeros((c))
    
    c=0
    for line in file:
        fields = line.strip().split()
        filename=fields[0]
        label=fields[1]
        #print (filename,label)
        img = Image.open(filename)

        ch=img.mode
        read=1
        if (ch=="RGB"):
            if (CH==1):
                print ("RGB-->Gray")
                img=img.convert('L')
        if (ch=="L"):
            if (CH==3):
                print ("========> Discarding gray image:"+filename)
                read=0
                
        if (read==1):
            if (RES=="resize"):
                img = img.resize((C,R), Image.ANTIALIAS)
            else:
                img=ImageOps.fit(img,(C,R), method=0, bleed=0.0, centering=(0.5, 0.5))
          
            X[c,:]=np.asarray( img, dtype="uint8" ).reshape(R,C,CH)
            L[c]=label
            c=c+1

    
    X=X[0:c,:]
    L=L[0:c]

    print(c,"images loaded")
    
    return X,L


