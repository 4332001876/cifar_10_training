class Config:
    # 模型路径及是否预训练
    SAVE_MODEL_PATH = 'tmp/cifar_10_model.pth'
    SAVE_FINAL_MODEL_PATH = 'tmp/cifar_10_final_model.pth'
    PRETRAINED_MODEL_PATH = 'tmp/cifar_10_model.pth'
    PRETRAINED = False

    # 数据集路径
    DATASET_PATH = 'tmp/cifar_10'

    # 训练参数
    EPOCHS = 12
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    
    STEP_LR_STEP_SIZE = 3
    STEP_LR_GAMMA = 0.5

    RANDOM_SEED = 10