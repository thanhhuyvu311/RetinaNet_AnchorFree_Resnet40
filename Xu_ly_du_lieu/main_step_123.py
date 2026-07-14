import xml.etree.ElementTree as ET
import os, glob
import pandas as pd

# [TILING - THÊM MỚI] import module tiling và thư viện PIL để crop ảnh
from tiling_utils import generate_tile_coords, get_bboxes_for_tile, crop_and_save_tile

if __name__ == '__main__':
    data_dir = '/home/huy/Documents/de_tai_tot_nghiep/Drone Thermal.v4i.voc'
    anno_dir = os.path.join(data_dir, 'anno')  # lay dia chi thuc muc anno
    img_dir = os.path.join(data_dir, 'imgs')


    tile_dir = os.path.join(data_dir, 'data_6_7_2026')
    os.makedirs(tile_dir, exist_ok=True)

    # [TILING - THÊM MỚI] thông số tiling — chỉnh ở đây nếu muốn thay đổi
    TILE_W   = 640
    TILE_H   = 512
    OVERLAP  = 0.20   # 20% chồng lấp

    # khoi tao dic gan nhan background voi id 0
    class_mapping = {'background': 0}

    # tao 1 mang de chua cac gia tri sau khi doc ra vao day
    # [TILING - ĐỔI] mỗi phần tử trong xml_list giờ là 1 tile, không còn là 1 ảnh gốc
    xml_list = []

    # lay dia chi file xml trong anno
    for xml_file in glob.glob(anno_dir + '/*.xml'):
        # lay vi tri cua file xml tren o dia
        tree = ET.parse(xml_file)
        # lay nut goc
        root = tree.getroot()

        # lay kich thuoc anh
        width_img  = int(root.find('size/width').text)
        height_img = int(root.find('size/height').text)
        path_img   = os.path.join(img_dir, root.find('filename').text)  # lay dia chi cung tung anh

        # Khoi tao danh sach rong cho tung anh de chua bndbox va class_id
        bboxes_of_img   = []
        class_ids_of_img = []

        # truy cap vao phan tu object de doc nhan, bbox
        for member in root.findall('object'):
            # tim ten class cua tung object
            class_name = member.find('name').text
            class_id   = class_mapping.setdefault(class_name, len(class_mapping))

            bndbox = member.find('bndbox')
            if bndbox is not None:
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                w = xmax - xmin
                h = ymax - ymin
                x = xmin + w / 2.0
                y = ymin + h / 2.0

                # BỘ LỌC CHẶN BÓNG MA: Chi lay nhung box co width va height > 0
                if w > 0 and h > 0:
                    bboxes_of_img.append([x, y, w, h])
                    class_ids_of_img.append(class_id)

        # ------------------------------------------------------------------
        # [TILING - THAY ĐỔI CHÍNH]
        # TRƯỚC: sau khi đọc xong object, append 1 dòng cho cả ảnh gốc
        # SAU:   chia ảnh thành N tiles, append N dòng (1 dòng/tile)
        # ------------------------------------------------------------------

        # Bước 1: Tạo tọa độ các tile
        tile_coords = generate_tile_coords(width_img, height_img, TILE_W, TILE_H, OVERLAP)

        for tile_idx, (tx, ty, tw, th) in enumerate(tile_coords):

            # Bước 2: Lọc và chuyển đổi bbox sang hệ tọa độ tile
            tile_bboxes, tile_cls_ids = get_bboxes_for_tile(
                bboxes_of_img, class_ids_of_img,
                tx, ty, tw, th,
                min_visibility=0.3
            )

            # Bước 3: Crop tile và lưu ra đĩa (trả về đường dẫn file tile)
            tile_path = crop_and_save_tile(path_img, tx, ty, tw, th, tile_dir, tile_idx)

            # Bước 4: Ghi 1 dòng CSV cho tile này (kể cả tile không có object)
            xml_list.append({
                'path_img' : tile_path,
                'bbox'     : tile_bboxes,
                'class_id' : tile_cls_ids
            })
        # ------------------------------------------------------------------

    # Tao DataFrame truc tiep tu list da duoc gom nhom
    grouped_df = pd.DataFrame(xml_list)

    # [TILING - ĐỔI TÊN] lưu ra file CSV mới để không ghi đè CSV gốc
    os.chdir(data_dir + '/csv_file/')
    grouped_df.to_csv('data_tile_6_7_2026.csv', index=False)
    #print(f"Xong! Đã tạo {len(xml_list)} tile records → data_information_tiled.csv")
    print(f"Ảnh tile được lưu tại: {tile_dir}")
