import aiohttp
import asyncio
import aiofiles
import os
import tqdm


class Downloader:
    def __init__(self, path, files_names, urls, semaphore=1000):
        self.sem = asyncio.Semaphore(semaphore)
        self.loop = asyncio.get_event_loop()
        self.path = path
        self.urls = urls
        # self.session = aiohttp.ClientSession()
        self.files_names = files_names
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.exited_files = set(os.listdir(self.path))

    async def _fetch(self, session, url, files_name):
        async with self.sem, session.get(url) as response:
            if response.status == 200:
                f = await aiofiles.open(f'{self.path}{files_name}', mode='wb')
                await f.write(await response.read())
                await f.close()

    async def _run(self):
        async with aiohttp.ClientSession() as session:
            for i, (url, files_name) in enumerate(tqdm.tqdm(zip(self.urls, self.files_names), total=len(self.urls))):
                if files_name not in self.exited_files:
                    # if i % 200 == 0:
                    #     await self.session.close()
                    #     self.session = aiohttp.ClientSession()
                    # async with self.session as session:
                    await asyncio.sleep(0.1)
                    await asyncio.ensure_future(self._fetch(session , url, files_name))
    
    def start(self):
        self.loop.run_until_complete(self._run())

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('unsplash-dataset/photos.tsv000', sep='\t', header=0)
    urls = df.photo_image_url.tolist()
    urls = [u+"?fm=jpg&w=224&h=224&fit=scale" for u in urls]
    files_names = df.photo_id.tolist()
    files_names = [n+".jpg" for n in files_names]
    downloader = Downloader("data_full/", files_names, urls, 1)
    downloader.start()