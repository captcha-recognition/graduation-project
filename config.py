# 数据配置项

## dataset 相关配置
train_data_path = "/Users/sjhuang/Documents/docs/dataset/captcha/train"
test_data_path = "/Users/sjhuang/Documents/docs/dataset/captcha/test"
train_rate = 0.8
# resize 大小
x = 32
y = 100
batch_size = 64


## cnn_rnn_ctc model params
crc_train_config = {
   "lr": 0.001,
   "momentum": 0.9,
   "epochs":  1000,
   "early_stop": 100,
   "map_to_seq_hidden": 64,
   "rnn_hidden": 256,
   "leaky_relu": False,
   "reload_checkpoint":"checkpoint/crc_checkpoints/",
   "show_interval":50,
   'valid_interval': 500,
   'save_interval': 2000,

}


## labels and chars only 英文和数字
CHARS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
num_class = len(LABEL2CHAR) + 1