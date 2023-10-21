import requests
import asyncio
from douyin_tiktok_scraper.scraper import Scraper

api = Scraper()

async def hybrid_parsing(url: str) -> dict:
    # Hybrid parsing(Douyin/TikTok URL)
    result = await api.hybrid_parsing(url)
    print(f"The hybrid parsing result:\n {result}")
    return result

def download_bilibili_video(bv_id: str):
    download_url = asyncio.run(hybrid_parsing(url=f"https://www.bilibili.com/video/{bv_id}/"))["video_url"]
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'}
    html = requests.get(download_url, headers=headers)
    save_path = f"./temp/bilibili_video/{bv_id}.mp4"
    with open(save_path, "wb") as f:
        f.write(html.content)
    return save_path