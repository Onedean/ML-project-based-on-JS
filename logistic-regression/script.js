import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { getData } from './data.js'

window.onload = async () => {
    const data = getData(400);  // 使用预先准备好的脚本生成二分类数据集

    tfvis.render.scatterplot(  // 数据可视化
        { name: '逻辑回归训练数据' },
        {
            values: [
                data.filter(p => p.label == 1),
                data.filter(p => p.label == 0),
            ]
        }
    );

    const model = tf.sequential();  // 初始化一个连续的神经网络模型
    model.add(tf.layers.dense({  // 为神经网络模型添加全连接层，其中神经元个数位1，tensor形状为2维
        units: 1,
        inputShape: [2],
        activation: 'sigmoid'  // 激活函数，将概率压缩至0到1
    }))

    model.compile({ loss: tf.losses.logLoss, optimizer: tf.train.adam(0.1) })  // 设置损失函数（采用对数损失），设置adam优化器（学习速率超参数为0.1）

    const inputs = tf.tensor(data.map(p => [p.x, p.y]));  // 将训练数据转换为tensor
    const labels = tf.tensor(data.map(p => p.label));

    await model.fit(inputs, labels, {  // 可视化拟合过程
        batchSize: 40,
        epochs: 50,
        callbacks: tfvis.show.fitCallbacks
            (
                { name: '训练过程' },
                ['loss']
            )
    });

    window.predict = (form) => {
        const pred = model.predict(tf.tensor([[form.x.value * 1, form.y.value * 1]]));
        alert(`预测结果：${pred.dataSync()[0]}`);
    }
};