import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getData } from './data.js';

window.onload = async () => {
    const data = getData(400);
    tfvis.render.scatterplot(
        { name: 'XOR训练数据集' },
        {
            values: [
                data.filter(p => p.label === 1),
                data.filter(p => p.label === 0)
            ]
        }
    );

    const model = tf.sequential();  // 初始化一个网络神经模型

    model.add(tf.layers.dense({  // 添加全连接层
        units: 4,
        inputShape: [2],
        activation: 'relu'  // 设置激活函数relu，非线性（若sigmod则无用，线性叠加线性仍然为线性）
    }));
    model.add(tf.layers.dense({  // 在添加一个全连接层
        units: 1,
        activation: 'sigmoid'
    }));

    model.compile({
        loss: tf.losses.logLoss,
        optimizer: tf.train.adam(0.1)
    });

    const inputs = tf.tensor(data.map(p => [p.x, p.y]));
    const labels = tf.tensor(data.map(p => p.label));

    await model.fit(inputs, labels, {
        epochs: 10,
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss']
        )
    });

    window.predict = (form) => {
        const pred = model.predict(tf.tensor([[form.x.value * 1, form.y.value * 1]]));
        alert(`预测结果：${pred.dataSync()[0]}`);
    };
};