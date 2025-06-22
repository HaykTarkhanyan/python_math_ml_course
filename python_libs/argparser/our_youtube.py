 
# դեմո անել pytubeի CLIը
import argparse
import logging
from pytube import YouTube

parser = argparse.ArgumentParser(description="Fetch YouTube video information using Pytube")
parser.add_argument("url", help="YouTube video URL")	
parser.add_argument("-d", "--detailed", action="store_true", help="Detailed video information")
parser.add_argument("-l", "--logging", help="Specify logging level", 
                    choices=["info", "error"], default="info")


args = parser.parse_args()
print(args)

url = args.url
detailed = args.detailed
level = args.logging

if level == "info":
    level = logging.INFO
elif level == "error":
    level = logging.ERROR



logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def get_video_info(url, detailed=False):
    logging.info(f"Fetching video information for {url}, detailed` {detailed}")
    try:
        yt = YouTube(url)
        print("Video Title:", yt.title)
        print("Video Views:", yt.views)
        if detailed:
            print("Video Author:", yt.author)
            print("Video Published Date:", yt.publish_date)
            print("Video Length:", yt.length, "seconds")
    except Exception as e:
        logging.exception(e)

get_video_info(url, detailed=detailed)


# https://www.youtube.com/watch?v=6_2ZJ4QW_O4
# https://www.youtube.com/watch?v=y4qtOVq_e0U
