import requests
from lxml import html
import os
import random
import time
import json

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0.2 Safari/605.1.15",
    "Mozilla/5.0 (Linux; Android 10; SM-G960F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Mobile Safari/537.36"
]

def fetch_bing_images(keyword, num_images=5000):
    if not os.path.exists(keyword):
        os.makedirs(keyword)

    headers = {
        "User-Agent": random.choice(USER_AGENTS)
    }

    count = 0
    start = 0
    while count < num_images:
        url = f"https://www.bing.com/images/search?q={keyword}&first={start}&count=35"
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        tree = html.fromstring(response.content)
        img_elements = tree.xpath('//a[@class="iusc"]/@m')

        if not img_elements:
            break  # 如果没有更多图片，退出循环

        for img in img_elements:
            if count >= num_images:
                break

            img_data = json.loads(img)
            img_source = img_data.get('murl')

            if img_source:
                try:
                    # 设置超时为 5 秒
                    img_content = requests.get(img_source, headers=headers, timeout=5).content
                    with open(os.path.join(keyword, f"{count + 1}.jpg"), 'wb') as img_file:
                        img_file.write(img_content)
                    # print(f"下载: {img_source}")
                    count += 1
                    print(count)
                except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                    print(f"下载失败: {img_source}，原因: {e}")

                # time.sleep(random.uniform(1, 3))  # 随机延迟

        start += 35  # 请求下一批图片

if __name__ == "__main__":
    keyword = input("请输入搜索关键词: ")
    fetch_bing_images(keyword)