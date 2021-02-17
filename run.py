import cv2
import os
import numpy as np
import threading
import time

NUM_CAMS = 1
INPUT_PATHS = [os.path.join(os.getcwd()+f'\sample_drive\cam_{i}\\') for i in range(NUM_CAMS)]
OUTPUT_PATH = os.path.join(os.getcwd()+f'\sample_drive\cam_output\\')

def get_imgs(dirs):
    imgs = []
    for d in dirs:
        #only getting the first 10 images from cam
        for p in os.listdir(d)[1000:1020]:
            path = os.path.join(d,p)
            temp = cv2.imread(path)
            imgs.append(temp)
    return imgs

def preprocess(images):
    imgs = []
    # convert RBG to grayscale
    for i in range(len(images)):
        cv2.imwrite(os.path.join(OUTPUT_PATH,str(i)+'_origin.jpg'),images[i])
        temp = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        # temp = cv2.equalizeHist(temp)
        # temp = cv2.blur(temp,(3,3))
        # ret, temp = cv2.threshold(temp,127,255,cv2.THRESH_BINARY)
        imgs.append(temp)
        cv2.imwrite(os.path.join(OUTPUT_PATH,str(i)+'.jpg'),temp)
    return imgs

def calc_avg_i(images):
    ret = images[0] * 1//len(images)

    for i in images:
        ret += i * 1//len(images)
    
    return ret

def calc_row(row, image, top50_avg_i, ret, pixel_count_map):
    for x in range(len(image[row])):
        if top50_avg_i[row][x] < image[row][x]:
            pixel_count_map[row][x] += 1
            ret[row][x] += image[row][x]

def calc_row_avg(row, image, pixel_count_map):
    for x in range(len(image[row])):
        image[row][x] //= pixel_count_map[row][x] 
        
def calc_avg_i0(images, avg_i):
    ret = np.zeros_like(avg_i) + 256
    top50_avg_i = avg_i * 1//2
    pixel_count_map = avg_i * 0
    
    num_threads = 127
    threads = []
    g = 0
    for i in images:
        g += 1
        print(g)
        for y in range(0,len(i),num_threads):
            threads = [threading.Thread(target=calc_row, args=(row,i,top50_avg_i,ret,pixel_count_map,)) for row in range(y,y+num_threads)]
            
            start = time.time()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            end = time.time()
            print(end - start)
            print(ret)
    
    ret -= 256

    for y in range(0,len(ret),num_threads):
        threads = [threading.Thread(target=calc_row_avg, args=(row,ret,pixel_count_map)) for row in range(y,y+num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    return ret

if __name__ == '__main__':
    # images = get_imgs(INPUT_PATHS)
    # processed_imgs = preprocess(images)
    # avg_image = calc_avg(processed_imgs)

    # adapt = cv2.adaptiveThreshold(avg_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    # adapt = cv2.bitwise_not(adapt)

    # cv2.imwrite(os.path.join(OUTPUT_PATH,'bitwise_not_adapt.jpg'),adapt)

    # kernel = np.ones((2,2),np.uint8)
    # erosion = cv2.erode(adapt,kernel,iterations = 1)
    # cv2.imwrite(os.path.join(OUTPUT_PATH,'erode.jpg'),erosion)
    # dilation = cv2.dilate(erosion,kernel,iterations = 1)
    # cv2.imwrite(os.path.join(OUTPUT_PATH,'dilation.jpg'),dilation)

    images = get_imgs(INPUT_PATHS)
    processed_imgs = preprocess(images)
    avg_i = calc_avg_i(processed_imgs)
    # print(avg_i)
    # print(avg_i + 256 + 250 - 256)
    avg_i0 = calc_avg_i0(processed_imgs, avg_i)
    cv2.imwrite(os.path.join(OUTPUT_PATH,'avg_i0.jpg'),avg_i0)
