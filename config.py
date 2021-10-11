# 数据配置项

## dataset 相关配置
train_data_path = "/Users/sjhuang/Documents/docs/dataset/captcha/train_data"
test_data_path = "/Users/sjhuang/Documents/docs/dataset/captcha/test"
train_rate = 0.8
# resize 大小
channel = 3
height = 32
weight = 100
batch_size = 64


## cnn_rnn_ctc model params
crc_train_config = {
   "lr": 1e-3,
   "momentum": 0.9,
   "epochs":  10000,
   "early_stop": 200,
   "map_to_seq_hidden": 64,
   "rnn_hidden": 256,
   "leaky_relu": False,
   "checkpoints_dir":"checkpoints/crc_checkpoints/",
   "reload_checkpoint": "checkpoints/crc_checkpoints/",
   "show_interval": 1,
   'valid_interval': 1,
   "drop_last":True,
   "num_workers":3,
   "decode_method":"beam_search",
   "beam_size":10,
   "name":"crc"
}


## labels and chars only 英文和数字
CHARS = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR2LABEL = {char: i for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
num_class = len(LABEL2CHAR)