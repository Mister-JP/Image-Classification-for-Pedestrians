def area(boxes,ratio):
    def checkin(coord, box):
        if coord[0] >= box[0] and coord[0] <= box[2]:
            if coord[1] >= box[1] and coord[1] <= box[3]:
                return True
        return False

    def getarea(box):
        return (box[0] - box[2]) * (box[1] - box[3])
    ret = []
    for box in boxes:
        #box = boxs[0]
        flag=0
        for check in range(len(ret)):
            if checkin([box[0],box[1]],ret[check]) and checkin([box[2],box[3]],ret[check]):
                flag=1
                break
            elif checkin([box[0],box[1]],ret[check]):
                if getarea(ret[check])/getarea([box[0],box[1],ret[check][2],ret[check][3]])>ratio:
                    flag = 1
                    break
            elif checkin([box[2],box[3]],ret[check]):
                if getarea(ret[check])/getarea([ret[check][0],ret[check][1],box[2],box[3]])>ratio:
                    flag=1
                    break
            elif checkin([ret[check][0], ret[check][1]], box) and checkin([ret[check][2], ret[check][3]], box):
                flag = 1
                if getarea(box) / getarea(ret[check]) > ratio:
                    #ret[check] = boxs
                    #ret.remove(check)
                    #ret.append(box)
                    break
        if flag==0:
            ret.append(box)
    return ret