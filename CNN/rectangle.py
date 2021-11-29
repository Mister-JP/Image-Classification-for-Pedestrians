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
        flag=0
        for check in ret:
            if checkin([box[0],box[1]],check) and checkin([box[2],box[3]],check):
                flag=1
                break
            elif checkin([box[0],box[1]],check):
                if getarea(check)/getarea([box[0],box[1],check[2],check[3]])>ratio:
                    flag = 1
                    break
            elif checkin([box[2],box[3]],check):
                if getarea(check)/getarea([check[0],check[1],box[2],box[3]])>ratio:
                    flag=1
                    break
            elif checkin([check[0],check[1]],box) and checkin([check[2],check[3]],box):
                flag = 1
                if getarea(box)/getarea(check)>ratio:
                    ret.remove(check)
                    ret.append(box)
                    break
        if flag==0:
            ret.append(box)
    return ret