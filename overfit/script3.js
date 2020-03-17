import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { getData } from './data.js'

window.onload = async () => {
    const data = getData(200, 3);  // 参数为样本数量和方差

    tfvis.render.scatterplot(
        { name: '训练数据' },
        {
            values: [
                data.filter(p => p.label === 1),
                data.filter(p => p.label === 0)
            ]
        }
    );

    const model = tf.sequential();

    model.add(tf.layers.dense({  // 添加复杂神经网络模型
        units: 1,
        inputShape: [2],
        activation: 'tanh',
        // kernelRegularizer:tf.regularizers.l2({l2:1}),  // 权重衰减法（模型的复杂度也成为损失的一部份）：l2正则法设置正则率为超参数1
    }))

    model.add(tf.layers.dropout({rate:0.9}));  // 丢弃法（将上一层神经元随机舍弃一部份权重）

    model.add(tf.layers.dense({  // 改为输出层
        units: 1,
        activation: 'sigmoid',
    }))

    model.compile({
        loss: tf.losses.logLoss,
        optimizer: tf.train.adam(0.1)
    })

    const inputs = tf.tensor(data.map(p => [p.x, p.y]));
    const labels = tf.tensor(data.map(p => p.label));

    await model.fit(inputs, labels, {
        validationSplit: 0.2,  // 分出20%数据作为验证集
        epochs: 200,
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss', 'val_loss'],
            { callbacks: ['onEpochEnd'] }
        )
    });

};