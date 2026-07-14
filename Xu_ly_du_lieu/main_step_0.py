import os,glob,shutil

if __name__ == "__main__":
    data_dir = '/home/huy/Documents/de_tai_tot_nghiep/Drone Thermal.v4i.voc'
    #--- lay dia chi folder ---
    train_data = os.path.join(data_dir,'train') #lay dia chi cua folder train
    valid_data = os.path.join(data_dir,'valid') #lay dia chi cua folder valid
    test_data = os.path.join(data_dir,'test') #lay dia chi cua folder test
    anno_dir = os.path.join(data_dir,'anno') #lay dia chi cua folder anno
    img_dir = os.path.join(data_dir,'imgs') #lay dia chi cua folder imgs

    #--- lay cac file xml va jpg
    """
    can di chuyen cac file trong folder train,test va valid sang folder anno va imgs
    thay doi folder bang cach thay doi ten dia chi folder o 2 dong code phia duoi
    """
    xml_file = glob.glob(os.path.join(train_data,'*.xml'))
    jpg_file = glob.glob(os.path.join(train_data,'*.jpg'))

    #--- di chuyen file sang folder khac
    #lay file name cua xml va jpg bang lenh os.path.basename
    for xml_path in xml_file:
        #co the xoa dau # o dong bien duoi de biet no lay gia tri nhu the nao
        #print(xml_path)
        xml_file_name = os.path.basename(xml_path)
        #print(xml_file_name)
        new_path = os.path.join(anno_dir,xml_file_name)
        #print(new_path)
        #di chuyen file
        shutil.move(xml_path,new_path)
        print(f"da di chuyen file xml thanh cong sang {new_path}")
    for jpg_path in jpg_file:
        jpg_file_name = os.path.basename(jpg_path)
        #print(jpg_file_name)
        new_path = os.path.join(img_dir,jpg_file_name)
        #print(new_path)
        #di chuyen file
        shutil.move(jpg_path, new_path)
        print(f"da di chuyen file jpg thanh cong sang {new_path}")