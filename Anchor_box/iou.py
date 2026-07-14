import tensorflow as tf
def convert_to_corners(boxes):
    """
    chyuen bdbox tu dang [x,y,w,h] sang [xmin,ymin,xmax,ymax]
    sau do noi cac gia tri nay thanh dang tensor
    """
    x_center,y_center,w,h = tf.split(boxes,4,axis=-1)
    """
    tach tung cot cua 1 mang thanh tensor 1 cot
    vd: 
    [50,50,100,100]
    [100,100,200,200] 
    -> sau khi bien doi se cho ra dang
    tf.tensor([
    [50]
    [100]],shape(2,1)
    """
    #print(x_center,y_center,w,h)
    x_min = x_center - (w/2.0)
    y_min = y_center - (h/2.0)
    x_max = x_center + (w/2.0)
    y_max = y_center + (h/2.0)
    return tf.concat([x_min,y_min,x_max,y_max],axis=-1)
def compute_iou(truth_box,predict_box):
    """
    tb = truth_box
    pb = predict box
    """
    #chuyen toa do
    box_tb = convert_to_corners(truth_box)
    box_pb = convert_to_corners(predict_box)

    #tach tao do lu,rd
    lu_box_tb = box_tb[:,:2] #lay tat ca hang, sau do lay 2 cot dau tien
    rd_box_tb = box_tb[:,-2:] #lay tat ca hang, sau do lay 2 cot cuoi cung
    lu_box_pb = box_pb[:,:2]
    rd_box_pb = box_pb[:,-2:]

    lu_box_tb = tf.expand_dims(lu_box_tb,0)
    rd_box_tb = tf.expand_dims(rd_box_tb,0)
    lu_box_pb = tf.expand_dims(lu_box_pb,1)
    rd_box_pb = tf.expand_dims(rd_box_pb,1)
    #tinh toa do phan giao nhau
    Intersection_lu = tf.math.maximum(lu_box_tb,lu_box_pb)
    Intersection_rd = tf.math.minimum(rd_box_tb,rd_box_pb)

    #wh cua phan giao
    Intersection_wh = tf.math.maximum(0.0,Intersection_rd-Intersection_lu) #neu khong giao thi = 0
    #print(Intersection_wh)
    #dien tich phan giao
    area_Intersection = Intersection_wh[:,:,0] * Intersection_wh[:,:,1]

    #dien tich phan gom (Union)
    area_box_tb = (rd_box_tb[:,:,0] - lu_box_tb[:,:,0]) * (rd_box_tb[:,:,1] - lu_box_tb[:,:,1])
    area_box_pb = (rd_box_pb[:,:,0] - lu_box_pb[:,:,0]) * (rd_box_pb[:,:,1] - lu_box_pb[:,:,1])

    area_Union = tf.math.maximum(area_box_tb+area_box_pb-area_Intersection,1e-8)

    #tinh Iou
    IoU = tf.math.divide_no_nan(area_Intersection,area_Union)

    return IoU
