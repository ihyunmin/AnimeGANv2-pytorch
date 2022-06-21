import cv2
import os

def main():
    input_path = os.path.join(os.getcwd(),'google_images_download', 'img')
    input_list = os.listdir(input_path)
    input_list.sort()

    os.makedirs(os.path.join(input_path, '..','polygon_256'), exist_ok=True)
    for i, path in enumerate(input_list):
        image_path = os.path.join(input_path, path)
        img = cv2.imread(image_path)
        h, w, c = img.shape
        minimum = min(h, w)
        top = max(int(h/2 - minimum/2), 0)
        btm = min(int(h/2 + minimum/2), h)
        rgt = min(int(w/2 + minimum/2), w)
        lef = max(int(w/2 - minimum/2), 0)
        test_img = img[ top:btm, lef:rgt, :]
        # cv2.imwrite(os.path.join(input_path,'..','polygon_256', str(i) + '.jpg'), test_img)
        img_256 = cv2.resize(test_img, dsize=(256,256), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(input_path,'..','polygon_256', str(i+367) + '.jpg'), img_256)
        print(img.shape)
    


if __name__=="__main__":
    main()