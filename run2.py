import cv2
import os
import numpy as np

NUM_CAMS = 5
INPUT_PATHS = [os.path.join(os.getcwd()+f'\sample_drive\cam_{i}\\') for i in range(NUM_CAMS)]
OUTPUT_PATH = [os.path.join(os.getcwd()+f'\sample_drive\cam_output\cam_{i}\\') for i in range(NUM_CAMS)]

def get_imgs(dirs, start, end):
    imgs = []

    for d in dirs:
        for p in os.listdir(d)[start:end]:
            path = os.path.join(d,p)
            temp = cv2.imread(path)
            imgs.append(temp)
    return imgs

def preprocess(images):
    imgs = []

    for i in range(len(images)):
        temp = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))
        temp = clahe.apply(temp)
        temp = cv2.GaussianBlur(temp,(3,3),0)
        imgs.append(temp)
    return imgs

def calc_avg_i(images):
    ret = images[0] * 1//len(images)
    for i in images:
        ret += i * 1//len(images)
    return ret

if __name__ == '__main__':
    for cam in range(NUM_CAMS):
        batch_size = 50
        num_of_images = len(os.listdir(INPUT_PATHS[cam]))
        batches = num_of_images // batch_size
        avg_image = None

        for i in range(1,batches):
            print(f'Cam_{cam}: {i/batches} %')
            start = (i * batch_size) - batch_size
            end = i * batch_size if i != batches - 1 else num_of_images

            images = get_imgs([INPUT_PATHS[cam]],start,end)
            processed_imgs = preprocess(images)

            temp_image = processed_imgs[0] * 1/num_of_images
            for i in processed_imgs:
                temp_image += i * 1/num_of_images
            try:
                if not avg_image:
                    avg_image = temp_image
            except:
                avg_image = cv2.add(avg_image,temp_image)

        avg_image = avg_image.astype('uint8')
        cv2.imwrite(os.path.join(OUTPUT_PATH[cam],'average.jpg'),avg_image)
        adapt = cv2.adaptiveThreshold(avg_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,151,15)
        cv2.imwrite(os.path.join(OUTPUT_PATH[cam],'adapt.jpg'),adapt)
        bitwise_not = cv2.bitwise_not(adapt)
        cv2.imwrite(os.path.join(OUTPUT_PATH[cam],'bitwise_not_adapt.jpg'),bitwise_not)
        erosion = cv2.erode(bitwise_not,np.ones((2,2),np.uint8),iterations = 1)
        dilation = cv2.dilate(erosion,np.ones((2,2),np.uint8),iterations = 5)
        cv2.imwrite(os.path.join(OUTPUT_PATH[cam],'final_mask.jpg'),dilation)
