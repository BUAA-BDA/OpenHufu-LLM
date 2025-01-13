import json


class Message(object):

    def __init__(self, msg_type, sender=-1, receiver=-1, seq=-1, timestamp=0, content=None):
        self.msg_type = msg_type
        self.sender = sender
        self.receiver = receiver
        self.seq = seq
        self.content = content
        self.timestamp = timestamp

    def count_bytes(self):
        from pympler import asizeof
        # download_bytes = asizeof.asizeof(self.content)
        # upload_cnt = len(self.receiver) if isinstance(self.receiver,
        #                                               list) else 1
        # upload_bytes = download_bytes * upload_cnt
        return asizeof.asizeof(self.content)

    def encode(self):
        # json
        json_msg = {
            'msg_type': self.msg_type,
            'sender': self.sender,
            'receiver': self.receiver,
            'content': self.content,
            'timestamp': self.timestamp,

        }
        return json.dumps(json_msg)

    def decode(self, json_str):
        # 不能像Bean那样有无参构造
        json_msg = json.loads(json_str)
        self.msg_type = json_msg['msg_type']
        self.sender = json_msg['sender']
        self.receiver = json_msg['receiver']
        self.state = json_msg['state']
        self.content = json_msg['content']
        self.timestamp = json_msg['timestamp']
        self.strategy = json_msg['strategy']

