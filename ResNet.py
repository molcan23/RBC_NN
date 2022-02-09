class CNNBlock(tf.keras.layers.Layer):

  def __init__(self, out_chanels, kernel_size=3):
    super(CNNBlock, self).__init__()
    self.conv = tf.keras.layers.Conv2D(out_chanels, len(SELECTED_COLUMNS), padding='same')
    self.bn = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.conv(input_tensor)
    x = self.bn(x, training=training)
    x = tf.nn.relu(x)

    return x
class ResBlock(tf.keras.layers.Layer):

  def __init__(self, channels):
    super(ResBlock, self).__init__()
    self.conn1 = CNNBlock(channels[0])
    self.conn2 = CNNBlock(channels[1])
    self.conn3 = CNNBlock(channels[2])

    self.pooling = tf.keras.layers.MaxPooling2D()
    self.identity_mapping = tf.keras.layers.Conv2D(channels[1], 1, padding='same')

  def call(self, input_tensor, training=False):
    x = self.conn1(input_tensor, training=training)
    x = self.conn2(x, training=training)
    x = self.conn3(
        x + self.identity_mapping(input_tensor, training=training),
        training=training
    )

    return self.pooling(x)


class ResNet_Like(tf.keras.Model):

  def __init__(self, number_of_outputs=1):
    super(ResNet_Like, self).__init__()
    self.block1 = ResBlock([32,64,128])
    self.block2 = ResBlock([128,128,256])
    self.block3 = ResBlock([128,256,512])

    self.pool = tf.keras.layers.GlobalAveragePooling2D()  # same as Flatten alegidly
    self.classifier = tf.keras.layers.Dense(number_of_outputs)

  def call(self, input_tensor, training=False):
    x = self.block1(input_tensor, training=training)
    x = keras.layers.Dropout(.2)(x)
    x = self.block2(x, training=training)
    x = keras.layers.Dropout(.2)(x)
    x = self.block3(x, training=training)
    x = self.pool(x)

    return self.classifier(x)

  def model(self):
    x = keras.Input(sape=(28,28,1))
    return keras.Model(inputs=[x], outputs=self.call(x))


model_3 = ResNet_Like(number_of_outputs=1)
EPOCHS = 1000

es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
mcp = ModelCheckpoint(filepath=f'{SAVE_PATH}/weights_ResNet.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard('logs')
# model_3.compile(
#     loss=keras.losses.MeanSquaredError(),
#     optimizer=tf.keras.optimizers.Adam(1e-4),
#     metrics=["mean_squared_error"]
# )

model_3.compile(
  loss=tf.keras.losses.MeanAbsoluteError(),
  optimizer=tf.keras.optimizers.Adam(1e-4),
  metrics=[tf.keras.metrics.MeanAbsoluteError()]
)

# model.summary()

history_3 = model_3.fit(
    X_train_CNN,
    y_train_CNN,
    shuffle=True,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=2,
    batch_size=64,
    callbacks=[es, rlr, mcp, tb]
)
