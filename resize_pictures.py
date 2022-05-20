import cv2
import os

def main():
    input_path = os.path.join(os.getcwd(),'dataset','train_ffhq')
    input_list = os.listdir(input_path)
    input_list.sort()

    os.makedirs(os.path.join(input_path, '..','train_ffhq_256'), exist_ok=True)
    for i, path in enumerate(input_list):
        image_path = os.path.join(input_path, path)
        img = cv2.imread(image_path)
        img_256 = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(input_path,'..','train_ffhq_256', path.split('.')[0] + '.jpg'), img_256)
        print(path)
    


if __name__=="__main__":
    main()