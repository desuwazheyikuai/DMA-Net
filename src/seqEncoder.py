from transformers import BertTokenizerFast, CLIPProcessor, AutoProcessor, Blip2Processor
import json


def _get_token(tokenIn):
    token = tokenIn.lower()
    return token


class SeqEncoder:
    def __init__(self, _config, JSONFile, textTokenizer=None):
        self.config=_config
        self.LEN_QUESTION = _config["LEN_QUESTION"]
        self.encoder_type = "answer"
        self.tokenizerName = textTokenizer
        self.textModel = _config["textModelPath"]
        self.clipList = _config["clipList"]
        if self.tokenizerName in self.clipList:
            self.tokenizer = CLIPProcessor.from_pretrained(self.textModel)
        elif self.tokenizerName in ["siglip_512"]:
            self.tokenizer = AutoProcessor.from_pretrained(self.textModel)
        elif self.tokenizerName in ["bert_base_uncased"]:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.textModel)
        Q_words = {}
#导入模块，有三个分词器，能够以三种不同的方式编码，构成三种词汇表，后两种的填充不需要特别设置
        with open(JSONFile) as json_data:
            self.data = json.load(json_data)["questions"]

        for i in range(len(self.data)):
            if self.data[i]["active"]:
                sentence = self.data[i]["question"]
                if sentence[-1] == "?" or sentence[-1] == ".":
                    sentence = sentence[:-1]
                tokens = sentence.split()
                for token in tokens:
                    token = _get_token(token)
                    if token not in Q_words:
                        Q_words[token] = 1
                    else:
                        Q_words[token] += 1
#遍历所有活跃的问题或者答案，分割开来，并且把这些词的出现次数以键值对的形式表现出来
        self.question_list_words = []
        self.question_words = {}

        sorted_words = sorted(Q_words.items(), key=lambda kv: kv[1], reverse=True)
        if self.tokenizerName in ["skipthoughts", "2lstm", "lstm"]:
            self.question_words = {"<EOS>": 0}
            self.question_list_words = ["<EOS>"]
            for i, (word, _) in enumerate(sorted_words):
                self.question_words[word] = i + 1
                self.question_list_words.append(word)
        elif self.tokenizerName in ["siglip_512"]:
            for i, (word, _) in enumerate(sorted_words):
                self.question_words[word] = self.tokenizer(
                    text=word, return_tensors="np"
                )["input_ids"][0][0]#返回一个单独的tokenid
                self.question_list_words.append(word)
        else:  # clip
            for i, (word, _) in enumerate(sorted_words):
                self.question_words[word] = self.tokenizer(text=word)["input_ids"][1]
                #为什么是1？也是为了得到一个tokenid
                self.question_list_words.append(word)
        pass
#用三种不同的方式构建词汇表，第一种类似于rsvqa,
    def encode(self, sentence, question=True):
        if sentence[-1] == "?" or sentence[-1] == ".":
            sentence = sentence[:-1]
        res = ''
        if self.tokenizerName in self.clipList or self.tokenizerName in [
            "bert_base_uncased"
        ]:
            if question:
                res = self.tokenizer(
                    text=sentence, padding="max_length", max_length=self.LEN_QUESTION
                )#检查 self.tokenizerName 是否在 self.clipList 或列表 ["bert_base_uncased"] 中
                return res
            #结构可能是：
            #    'input_ids': numpy.ndarray(shape=(1, self.LEN_QUESTION)),
             #   'attention_mask': numpy.ndarray(shape=(1, self.LEN_QUESTION)),
              #  'token_type_ids': numpy.ndarray(shape=(1, self.LEN_QUESTION))  # 如果分词器支持的话

        elif self.tokenizerName in ["siglip_512"]:
            if question:
                res = self.tokenizer(
                    text=sentence,
                    padding="max_length",
                    max_length=self.LEN_QUESTION,
                    return_tensors="np",
                )
                return res
            #结构可能是：input_ids': numpy.ndarray(shape=(1, self.LEN_QUESTION)),
    #'attention_mask': numpy.ndarray(shape=(1, self.LEN_QUESTION)

        elif self.tokenizerName in ["skipthoughts", "2lstm", "lstm"]:
            res = []
            if sentence[-1] == "?" or sentence[-1] == ".":
                sentence = sentence[:-1]
            tokens = sentence.split()
            for token in tokens:
                token = _get_token(token)
                res.append(self.question_words[token])

            if question:
                res.append(self.question_words["<EOS>"])
                while len(res) < self.LEN_QUESTION:
                    res.append(self.question_words["<EOS>"])
                res = res[: self.LEN_QUESTION]
        else:
            res = "unexpected wrong"
        return res
#用几种不同的方式编码，设置了填充长度，使转化后的结果达到预设的长度，
    def getVocab(self, question=True):
        if question:
            return self.question_list_words
