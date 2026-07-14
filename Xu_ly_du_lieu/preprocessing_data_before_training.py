"""
file code nay se xu ly file csv da tao ra, de chyuen cac du lieu so trong file csv tu dang chuoi  thanh dang so co the tinh toan duoc
"""
import ast
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def change_string2number(input_csv):
    """
    chuyen gia tri string sang list
    :param input_csv:
    :return:
    """
    data_frame = pd.read_csv(input_csv)
    data_frame['bbox'] = data_frame['bbox'].apply(ast.literal_eval)
    data_frame['class_id'] = data_frame['class_id'].apply(ast.literal_eval)
    return data_frame
def create_dataset_from_dataframe(data_frame):
    """
    vi nhan dien da vat the nen cac mang 2 chieu co kich thuoc khong dong deu nhau,
    nen khi chuyen chung thanh dang tensor, chung se co kich thuoc khong dong bo, su dung
    ham ragged tensor de giai quyet van de
    :param data_frame:
    :return:
    """
    tensor_bbox = tf.ragged.constant(data_frame['bbox'].tolist(),ragged_rank=1)
    tensor_class_id = tf.ragged.constant(data_frame['class_id'].apply(lambda x: [i - 1 for i in x]).tolist())
    tensor_path = tf.constant(data_frame['path_img'].tolist())
    #print(tensor_bbox[0:2],tensor_class_id[0:2],tensor_path) #kiem tra xem no in ra cai gi khi chuyen doi

    #dua 3 khoi tensor vao  tensor slice
    dataset = tf.data.Dataset.from_tensor_slices((tensor_path,tensor_bbox,tensor_class_id))
    return dataset
def read_img_and_label(path,bbox,class_id):
    #doc file tu o cung
    img = tf.io.read_file(path)
    #gia ma anh
    img = tf.image.decode_jpeg(img,channels=3)
    #ep kieu du lieu
    img = tf.cast(img,tf.float32)
    return img,bbox,class_id
def resize_and_pad_img(img,bbox,class_id,target_size):
    """
    ham nay nhan vao 1 buc anh (dc dong bang decode_jpg)
    cung voi bbox va class id
    """
    #lay chieu cao va chieu rong
    shape = tf.cast(tf.shape(img)[:2],dtype=tf.float32)
    img_h = shape[0]
    img_w = shape[1]

    ratio = tf.cast(target_size,tf.float32) / tf.math.maximum(img_h,img_w) #tinh ti le anh

    new_img_h = tf.cast(img_h*ratio,tf.int32)
    new_img_w = tf.cast(img_w*ratio,tf.int32)

    #resize anh
    img = tf.image.resize(img,[new_img_h,new_img_w])

    img = tf.image.pad_to_bounding_box(
        img,
        offset_height=0,
        offset_width=0,
        target_height=target_size,
        target_width=target_size
    )

    #co gian bdbox
    bbox = ratio*bbox
    return img,bbox,class_id

def data_augment(img, bbox, class_id):
    """
    Data Augmentation: Tăng cường dữ liệu chuyên biệt cho ảnh nhiệt từ drone
    """
    target_size = tf.cast(tf.shape(img)[1], tf.float32)
    
    # Random Lật ngang (Horizontal Flip)
    # Rất hiệu quả cho ảnh drone chụp từ trên xuống, giúp mô hình bớt phụ thuộc vào hướng của người
    flip_cond = tf.random.uniform([]) > 0.5
    img = tf.cond(flip_cond, lambda: tf.image.flip_left_right(img), lambda: img)
    
    def flip_boxes(b):
        x_center, y_center, w, h = tf.split(b, 4, axis=-1)
        x_center = target_size - x_center
        return tf.concat([x_center, y_center, w, h], axis=-1)
    
    bbox = tf.cond(flip_cond, lambda: flip_boxes(bbox), lambda: bbox)
    
    # Random Contrast (Tương phản): Rất tốt cho ảnh nhiệt vì đôi khi người và nền có nhiệt độ gần nhau
    img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
    
    # Random Brightness (Độ sáng)
    img = tf.image.random_brightness(img, max_delta=0.15)
    
    img = tf.clip_by_value(img, 0.0, 255.0)
    
    return img, bbox, class_id


def visualize_data(img_tensor,bbox_tensor,ax,title):
    #ep kieu anh ve uint8 de plt doc dc
    img_array = tf.cast(img_tensor,tf.uint8).numpy()
    ax.imshow(img_array)
    ax.set_title(title)
    ax.axis('off')

    for box in bbox_tensor.numpy():
        x_center , y_center , w ,h = box
        x_min = x_center - (w/2)
        y_min = y_center - (h/2)

        #ve box
        #REctangle lay gia tri x_min y_min w h
        rect = patches.Rectangle(
            (x_min,y_min),w,h,
            linewidth=2,edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

"""
if __name__ == '__main__':
    train_data_csv = '/home/huy/Documents/de_tai_tot_nghiep/object_detect/csv_file/train_data.csv'
    data = change_string2number(train_data_csv)
    data_set = create_dataset_from_dataframe(data)

    data_set_raw = data_set.map(read_img_and_label,num_parallel_calls=tf.data.AUTOTUNE) #num parallel su dung nhieu luong cpu

    data_set_resized = data_set_raw.map(lambda img,box,cls: resize_and_pad_img(img,box,cls,target_size=224),num_parallel_calls=tf.data.AUTOTUNE)


    for (raw_img,raw_box,raw_cls),(res_img,res_box,res_cls) in tf.data.Dataset.zip((data_set_raw,data_set_resized)).take(11):
        fig,ax = plt.subplots(1,2,figsize = (10,7))
        #hien thi anh
        visualize_data(raw_img,raw_box,ax[0],'anh goc')
        visualize_data(res_img,res_box,ax[1], 'anh res')

        plt.tight_layout()
        plt.show()
"""



