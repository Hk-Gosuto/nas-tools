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
        if self._provider == "ollama":
            if self._api_url:
                self._ollama_client = Client(host=self._api_url)
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
format: {"cn_title":string|null,"en_title":string|null,"jp_title":string|null,"version":string|null,"part":string|null,"year":string|null,"resolution":string,"season":number|null,"episode":number|null,"subtitle_group":string,"subtitle_language":string|null}
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
