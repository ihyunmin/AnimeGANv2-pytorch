from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import time
import os
import urllib
import numpy as np
import cv2
import requests

#
# search keyword 설정 및 
#
driver_path = os.path.join(os.getcwd(),'chromedriver')
driver = webdriver.Chrome(driver_path) # 웹드라이버 파일 경로
search_keyword = 'something'
url_keyword = urllib.parse.quote(search_keyword)
driver.get(f"https://www.google.com/search?q={url_keyword}&hl=ko&sxsrf=ALiCzsZEbWNAhhSAWpb3sDGdubrOAA3hRQ:1654741887622&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiNiqDzqZ_4AhUKDd4KHeehCTQQ_AUoAXoECAEQAw&biw=1080&bih=4500&dpr=1") # 이미지 검색 URL
time.sleep(5) # 5초 동안 페이지 로딩 기다리며 파이썬은 쉰다. 

#
# Scroll 하는 구간
#
for i in range(100):
    driver.find_element_by_tag_name('body').send_keys(Keys.PAGE_DOWN)
    time.sleep(0.1) # 5초 동안 페이지 로딩 기다리며 파이썬은 쉰다.
time.sleep(5) # 5초 동안 페이지 로딩 기다리며 파이썬은 쉰다.
driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div[1]/div[2]/div[2]/input').click()

for i in range(100):
    driver.find_element_by_tag_name('body').send_keys(Keys.PAGE_DOWN)
time.sleep(5) # 5초 동안 페이지 로딩 기다리며 파이썬은 쉰다.


#
# Image download Link 탐색 구간
# 
images = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd")
print(search_keyword+' 찾은 이미지 개수:',len(images))

links=[]
for i in range(1,len(images)):
    try:
        driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div['+str(i)+']/a[1]/div[1]/img').click()
        link = driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img').get_attribute('src')
        links.append(link)
        driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[2]/a').click()
        print(search_keyword+' 링크 수집 중..... number :'+str(i)+'/'+str(len(images)))
    except:
        continue
    
#
# Image download 구간
# 
forbidden=0
for k,i in enumerate(links):
    try:
        url = i
        start = time.time()
        # resp = urllib.urlopen(url)
        # image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
        image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
        cv2.imwrite("./img/"+search_keyword.replace(' ','')+'_'+str(k-forbidden)+".jpg", image)
        # urllib.request.urlretrieve(url, "./img/"+str(k-forbidden)+".jpg")
        print(str(k+1)+'/'+str(len(links))+' '+search_keyword+' 다운로드 중....... Download time : '+str(time.time() - start)[:5]+' 초' + 'file name : ' + str(k-forbidden)+".jpg")
    except:
        forbidden+=1
        continue
print(search_keyword+' ---다운로드 완료---')

driver.close()