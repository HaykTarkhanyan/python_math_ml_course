import json
import time
from tqdm import tqdm
from pytubefix import Playlist, YouTube

def get_info(url: str) -> dict:
    """Fetches basic info about a YouTube video.
    
    Note:
        Returns 1. title, 2. views, ....
    
    Args:
        url (str): The URL of the YouTube video.
    
    Returns:
        dict: A dictionary containing the video's title, author, and description.
    """
    yt = YouTube(url)
    
    data = {
        "title": yt.title,
        "views": yt.views,
        "length": yt.length,
        # "publish_date": yt.publish_date#time.strftime("%Y-%m-%d", time.localtime(yt.publish_date)),
    }
    
    return data

def get_playlist_info(pl_url: str) -> list:
    """Fetches basic info about a YouTube playlist.
    
    Args:
        pl_url (str): The URL of the YouTube playlist.
        
    Returns:
        dict: A dictionary containing the playlist's title, author, and description.
    """
    print(f"Fetching info for playlist: {pl_url}")	
    
    pl = Playlist(pl_url)
    
    all_videos = []
    for video in tqdm(pl.video_urls):
        print(f"Fetching info for video: {video}")
        all_videos.append(get_info(video))

    all_videos.sort(key=lambda x: x["views"], reverse=True)
    
    return all_videos

if __name__ == "__main__":
    st = time.perf_counter()
    playlist_url = input("Enter the YouTube playlist URL: ")	
    
    playlist_info = get_playlist_info(playlist_url)
    print(playlist_info)
    end = time.perf_counter()
    
    with open("playlist_info.json", "w") as f:
        json.dump(playlist_info, f, indent=4)   
    print(f"Time taken: {end - st:.2f} seconds")