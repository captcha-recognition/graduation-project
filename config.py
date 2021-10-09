# 数据配置项

## dataset 相关配置
train_data_path = "/Users/sjhuang/Documents/docs/dataset/captcha/train"
test_data_path = "/Users/sjhuang/Documents/docs/dataset/captcha/test"
train_rate = 0.8
# resize 大小
x = 30
y = 90
batch_size = 64


## cnn_rnn_ctc model params
crc_lr = 0.001
crc_momentum = 0.9
crc_epochs = 1000
crc_early_stop = 100
crc_name = f"crc_{crc_lr}_{crc_momentum}_{crc_epochs}_{crc_early_stop}"
