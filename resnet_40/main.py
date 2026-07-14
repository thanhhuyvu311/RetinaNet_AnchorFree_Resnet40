import tensorflow as tf


def Conv_block(x, filter, stride):
    """
    x: input
    filter: so luong filter
    stride: buoc nhay
    Day la block lam thay doi kich thuoc input (chi giam hoac khong giam)
    input va output se khong khop nhau, sau khi qua ham nay, input va output se dc can bang so luong kenh de co the cong voi nhau
    """
    x_skip = x
    f1, f2 = filter

    # block dau tien
    #neu filter stride la 1 thi chi la buoc nhay 1, neu stride la 2 thi no dai dien cho viec maxpool
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(1, 1), strides=(stride, stride), padding='valid',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # block thu 2
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # block thu 3 (tang so kenh ban do dac trung)
    x = tf.keras.layers.Conv2D(filters=f2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # tang so kenh x_skip de cong vao x tai block 3
    x_skip = tf.keras.layers.Conv2D(filters=f2, kernel_size=(1, 1), strides=(stride, stride), padding='valid',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_skip)
    x_skip = tf.keras.layers.BatchNormalization()(x_skip)

    # cong x_skip vao x
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.ReLU()(x)

    return x #(so kenh ban do dac trung = f2)


def Res_id_block(x, filter):
    """
    resnet block khong thay doi kich thuoc
    khoi nay chi hoat dong khi input va output co cung kich thuoc [w,h,c]
    """
    x_skip = x #(so kenh x skip = f2 vi la return cua ham conv_block)
    f1, f2 = filter

    # block 1
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(1, 1), padding='valid',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # block 2
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(3, 3), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # block 3
    x = tf.keras.layers.Conv2D(filters=f2, kernel_size=(1, 1), padding='valid',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # add
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.ReLU()(x)
    return x


def resnet_40_backbone():
    """
    ResNet-40 backbone: giong ResNet-50 nhung bo block 5 (P5/C5, Stride 32).
    Chi xuat ra 2 feature map:
      - block_3_out: C3 (Stride 8)  -> P3 trong FPN
      - block_4_out: C4 (Stride 16) -> P4 trong FPN
    """
    # input size
    input_dim = tf.keras.layers.Input(shape=(None, None, 3))

    # su dung padding de giu lai bien anh truoc khi vao conv2d 7x7
    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='padding_zero')(input_dim)

    # khoi 1
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid', use_bias=False,
                               name='Conv7x7')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='overlapping')(x)

    # khoi 2
    x = Conv_block(x, (64, 256), 1)   # 3 layers
    x = Res_id_block(x, (64, 256))    # 3 layers
    x = Res_id_block(x, (64, 256))    # 3 layers

    # khoi 3 -> C3 / P3 (Stride 8)
    x = Conv_block(x, (128, 512), 2)  # 3 layers
    x = Res_id_block(x, (128, 512))   # 3 layers
    x = Res_id_block(x, (128, 512))   # 3 layers
    x = Res_id_block(x, (128, 512))   # 3 layers
    block_3_out = x

    # khoi 4 -> C4 / P4 (Stride 16)
    x = Conv_block(x, (256, 1024), 2) # 3 layers
    x = Res_id_block(x, (256, 1024))  # 3 layers
    x = Res_id_block(x, (256, 1024))  # 3 layers
    x = Res_id_block(x, (256, 1024))  # 3 layers
    x = Res_id_block(x, (256, 1024))  # 3 layers
    x = Res_id_block(x, (256, 1024))  # 3 layers
    block_4_out = x

    # khoi 5 (P5) da bi xoa trong ResNet-40

    model = tf.keras.models.Model(
        inputs=input_dim,
        outputs=[block_3_out, block_4_out],
        name='resnet_40'
    )
    return model


def head_of_model(output_filters, bias_init):
    """
    xay dung dau ra cho du doan class va bbox
    """
    head = tf.keras.Sequential([tf.keras.layers.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer=kernel_init))
        head.add(tf.keras.layers.ReLU())
    head.add(tf.keras.layers.Conv2D(output_filters, 3, 1, padding='same', kernel_initializer=kernel_init,
                                    bias_initializer=bias_init))
    return head
