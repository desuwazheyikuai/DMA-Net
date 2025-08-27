import json


class answerNumber:
    def __init__(self, _config, JSONFile):
        self.config = _config
        with open(JSONFile) as json_data:
            self.answers = json.load(json_data)['answers']
#读取 JSON 文件中的内容，并提取 ‘answers’ 键对应的全部值。

    def encode(self, qType, answer):
        output = self.answers[int(qType)-1]['answer'][answer]#获得答案的编码数值,比如说no对应的编码是0
        return output
#目的就是根据问题类型找到对应的多个答案
