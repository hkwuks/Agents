import requests
import os
from tavily import TavilyClient
from loguru import logger
import re


def get_weather(city: str) -> str:
    '''
    通过wttr.in API查询真实的天气信息
    Args:
        city: 城市

    Returns:
        该城市的实时天气状况
    '''

    url = f'https://wttr.in/{city}?format=j1'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_c']

        return f'{city}当前天气：{weather_desc}，气温{temp_c}摄氏度'
    except requests.exceptions.RequestException as e:
        return f'错误：查询天气时遇到网络问题 - {e}'
    except (KeyError, IndexError) as e:
        return f'错误：解析天气数据错误，可能是城市名无效 - {e}'


def get_attraction(city: str, weather: str) -> str:
    '''
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    Args:
        city:
        weather:

    Returns:
         景点推荐
    '''
    api_key = os.environ.get('TAVILY_API_KEY')
    if not api_key:
        return '错误：未配置TAVILY_API_KEY环境变量'

    tavily = TavilyClient(api_key)

    query = f"{city}在{weather}天气下最值得去的旅游景点推荐及理由"

    try:
        response = tavily.search(query, search_depth='basic', include_answer=True)

        if response.get('answer'):
            return response['answer']

        formatted_results = []
        for result in response.get('results', []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐。"
        return "根据搜索，为您找到以下信息：\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"错误：执行Tavily搜索时出现问题 - {e}"


AGENT_SYSTEM_PROMPT = '''
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用的工具一步一步地解决问题。

# 可用工具：
- `get_weather(city: str)`:  查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 行动格式：
你的回答必须严格遵循以下格式。首先是你的思考过程，然后是你要执行的具体行动。
Thought: [这里是你的思考过程和下一步计划]
Action: [这里是你要调用的工具，格式为 function_name(arg_name="arg_value")]

# 任务完成：
当你收集到足够的信息，能够回答用户的最终问题时，你必须使用`finish(answer="...")`来输出最终答案。

请开始吧！
'''
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction
}

from openai import OpenAI


class OpenAICompatibleClient:
    '''
    一个用于调用任何兼容OpenAI接口的LLM服务客户端
    '''

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        '''
        调用LLM API来生成回答
        Args:
            prompt:
            system_prompt:

        Returns:
            str: 回答
        '''
        logger.info("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            logger.info("大语言模型响应成功。")
            return answer
        except Exception as e:
            logger.error(f'调用LLM API时发生错误 - {e}')
            return "错误：调用大语言模型服务时出错。"


def main():
    # 1. 配置LLM
    API_KEY = ""
    BASE_URL = ""
    MODEL_ID = ""
    TAVILY_API_KEY = ""

    llm = OpenAICompatibleClient(MODEL_ID, API_KEY, BASE_URL)

    # 2. 初始化
    user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
    prompt_history = [f'用户请求：{user_prompt}']

    logger.info(f'用户输入：{user_prompt}\n' + '=' * 40)

    # 3. 主循环
    for i in range(5):
        logger.info(f'--- 循环{i + 1} ---\n')

        full_prompt = '\n'.join(prompt_history)

        llm_out = llm.generate(prompt=full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
        logger.info(f'模型输出：\n{llm_out}\n')

        prompt_history.append(llm_out)

        action_match = re.search(r'Action: (.*)', llm_out, re.DOTALL)
        if not action_match:
            logger.error("解析错误：模型输出中未找到Action。")
            break

        action_str = action_match.group(1).strip()

        if action_str.startswith("finish"):
            final_answer = re.search(r'finish\(answer="(.*)"\)', action_str).group(1)
            logger.info(f'任务完成，最终答案：{final_answer}')
            break

        tool_name = re.search(r"(\w+)\(", action_str).group(1)
        args_str = re.search(r"\((.*)\)", action_str).group(1)
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

        if tool_name in available_tools:
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误：未定义的工具 {tool_name}"

        observation_str = f'Observation: {observation}'
        logger.info(f'{observation_str}\n' + '=' * 40)
        prompt_history.append(observation_str)
