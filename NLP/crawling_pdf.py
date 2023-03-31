from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
from random import randrange
from bs4 import BeautifulSoup
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
import sys
from datetime import datetime

url = f''
driver = webdriver.Chrome(ChromeDriverManager().install())

# 기본
pdf_links = []
for link in driver.find_elements_by_xpath("//a"):
    url = link.get_attribute("href")
    if url.endswith(".pdf"):
        pdf_links.append(url)
print(pdf_links)

driver.close()

# href 링크 이상한 경우
url = f"https://dream.kotra.or.kr/kotranews/cms/indReport/actionIndReportDetail.do?SITE_NO=3&MENU_ID=280&CONTENTS_NO=1&pHotClipTyName=DEEP&pRptNo=13576"
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get(url)
time.sleep(1)

a_tags = driver.find_elements(By.TAG_NAME, 'a')
for a_tag in a_tags:
    if a_tag.text[-4:] == '.pdf':
        a_tag.click()
        time.sleep(1)
        driver.implicitly_wait(10)
        print(a_tag)

driver.close()