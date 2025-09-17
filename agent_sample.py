import requests
import json

AGENT_SYSTEM_PROMPT='''
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

def get_weather(city: str)-> str:
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
