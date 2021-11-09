class read2Dobj:
    def __init__(self, objclass = None, objid=None, conf=None, coord=None):
        self.objclass = objclass
        self.objid = objid
        self.conf = conf
        self.coord = coord
    def setobjclass(self, objclass):
        self.objclass = objclass
    def setobjid(self, objid):
        self.objid = objid
    def setconf(self,conf):
        self.conf = conf
    def setcoord(self, coord):
        self.coord = coord
class readImage:
    count = 0
    def __init__(self,name=None, width=None, height=None, numobj=None):
        self.name = name
        self.width = width
        self.height = height
        self.numobj = numobj
    def setname(self,name):
        self.name = name
    def setwidth(self,width):
        self.width = width
    def setheight(self,height):
        self.height = height
    def setnumobj(self,numobj):
        self.numobj = numobj
        self.objs = [read2Dobj() for i in range(int(numobj))]
    def addobjs(self, objclass=None, objid=None, conf=None, coord=None):
        self.objs[self.count].setobjclass(objclass)
        self.objs[self.count].setobjid(objid)
        self.objs[self.count].setconf(conf)
        self.objs[self.count].setcoord(coord)
        self.count+=1
class readSequences:
    count = 0
    def __init__(self, seq_id, path_to_data, numimages):
        self.seq_id = seq_id
        self.path_to_data = path_to_data
        self.numimages = numimages
        self.imgs = [readImage() for i in range(int(self.numimages))]
    def addimage(self, name, width, height, numobj):
        self.imgs[self.count].setname(name)
        self.imgs[self.count].setwidth(width)
        self.imgs[self.count].setheight(height)
        self.imgs[self.count].setnumobj(numobj)
        self.count+=1
    def addobj(self, objclass=None, objid=None, conf=None, coord=None):
        self.imgs[self.count-1].addobjs(objclass, objid,conf,coord)
