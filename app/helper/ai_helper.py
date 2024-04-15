import json

import openai
from ollama import Client

from app.utils.commons import singleton
from config import Config

import log


@singleton
class AiHelper:
    _api_key = None
    _api_url = None
    _model = None
    _provider = None
    _ollama_client = None

    def __init__(self):
        self.init_config()

    def init_config(self):
        self._model = Config().get_config("ai").get("model")
        self._provider = Config().get_config("ai").get("provider")
        self._api_key = Config().get_config("ai").get("api_key")
        self._api_url = Config().get_config("ai").get("api_url")
        if self._provider == "openai":
            if not self._model:
                self._model = "gpt-3.5-turbo-0125"
            if self._api_key:
                openai.api_key = self._api_key
            if self._api_url:
                openai.base_url = self._api_url + "/v1/"
            else:
                proxy_conf = Config().get_proxies()
                if proxy_conf and proxy_conf.get("https"):
                    openai.proxy = proxy_conf.get("https")
        else:
            if self._api_url:
                _ollama_client = Client(host=self._api_url)
            if not self._api_key:
                self._api_key = "ollama"
            if not self._model:
                self._model = self._ollama_client.list()["models"][0]["name"]

    def get_state(self):
        return True if self._api_key else False

    @staticmethod
    def __get_ollama_model(client, model, message, prompt=None, **kwargs):
        """
        获取模型
        """
        if not isinstance(message, list):
            if prompt:
                message = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message},
                ]
            else:
                message = [{"role": "user", "content": message}]
        completion = client.chat(model=model, messages=message, **kwargs)
        return completion["message"]["content"]

    @staticmethod
    def __get_openai_model(model, message, prompt=None, **kwargs):
        """
        获取模型
        """
        if not isinstance(message, list):
            if prompt:
                message = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message},
                ]
            else:
                message = [{"role": "user", "content": message}]
        completion = openai.chat.completions.create(
            model=model,
            messages=message,
            **kwargs,
        )
        return completion.choices[0].message.content

    @staticmethod
    def __get_openai_json_model(message, prompt=None, **kwargs):
        """
        获取模型
        """
        if not isinstance(message, list):
            if prompt:
                message = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message},
                ]
            else:
                message = [{"role": "user", "content": message}]
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            response_format={"type": "json_object"},
            messages=message,
            **kwargs,
        )
        return completion.choices[0].message.content

    def get_media_name(self, filename):
        """
        从文件名中提取媒体名称等要素
        :param filename: 文件名
        :return: Json
        """
        if not self.get_state():
            return None
        result = ""
        try:
            _filename_prompt = """
I will give you a movie/tvshow file name.You need to return a Json.
format: {"cn_title":string|null,"en_title":string|null,"jp_title":string|null,"version":string|null,"part":string|null,"year":string|null,resolution":string,"season":number|null,"episode":number|null,"subtitle_group":string,"subtitle_language":string|null}

example:
[桜都字幕组] 无职转生～到了异世界就拿出真本事～ S2 / Mushoku Tensei S2  [13][1080p@60FPS][繁体内嵌]
{"cn_title":"无职转生～到了异世界就拿出真本事～","en_title":"Mushoku Tensei","resolution":"1080","season":2,"episode":13,"subtitle_group":"桜都字幕组","subtitle_language":"繁体"}
[Billion Meta Lab] 终末列车寻往何方 Shuumatsu Torein Dokoe Iku [01][1080][简日内嵌]
{"cn_title":"终末列车寻往何方","en_title":"Shuumatsu Torein Dokoe Iku","resolution":"1080","episode":1,"subtitle_group":"Billion Meta Lab","subtitle_language":"简日"}
【幻樱字幕组】【1月新番】【迷宫饭 Dungeon Meshi】【15】【BIG5_MP4】【1920X1080】
{"cn_title":"迷宫饭","en_title":"Dungeon Meshi","resolution":"1080","episode":15,"subtitle_group":"幻樱字幕组","subtitle_language":"BIG5"}
【旋风字幕组】死神Bleach 第355话「死神参战 ！静灵庭正月特辑！」MP4/简体 1280x720
{"cn_title":"死神","en_title":"Bleach","resolution":"1280x720","episode":355,"subtitle_group":"旋风字幕组","subtitle_language":"简体"}
"""
            result = None
            if self._provider == "openai":
                if self._model == "gpt-3.5-turbo-0125":
                    result = self.__get_openai_json_model(
                        prompt=_filename_prompt, message=filename
                    )
                else:
                    result = self.__get_openai_model(
                        prompt=_filename_prompt, message=filename, model=self._model
                    )
            else:
                result = self.__get_ollama_model(
                    prompt=_filename_prompt,
                    message=filename,
                    model=self._model,
                    client=self._ollama_client,
                )
            log.info(f"[AI] {result}")
            return json.loads(result)
        except Exception as e:
            print(f"{str(e)}：{result}")
            return {}
