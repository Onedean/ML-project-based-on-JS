import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { getIrisData, IRIS_CLASSES } from './data.js'  // 导入获取训练集和验证集及输出中文类别的方法
import { sigmoid, softmax } from '@tensorflow/tfjs';

window.onload = async () => {
    // 获取15%的数据用于验证集，剩余75%用于训练集
    const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);  // 参数分别为训练集的所有特征,训练集的所有标签,验证集的所有特征,验证集的所有标签
    /* xTrain.print();
    yTrain.print();
    xTest.print();
    xTest.print();
    console.log(IRIS_CLASSES); */

    const model = tf.sequential();

    model.add(tf.layers.dense({
        units: 10,  // 凭直觉设置神经元个数为10，后期训练看调整效果
        inputShape: [xTrain.shape[1]],  // 特征为4的一维数组
        activation: 'sigmoid'  // 激活函数带来非线性变换即可
    }))

    model.add(tf.layers.dense({
        units: 3,  // 第二层神经元个数必须为输出类别的个数
        activation: 'softmax'  // 为了输出多概率且和为1问题，使用softmax激活函数
    }))

    model.compile({
        loss: 'categoricalCrossentropy',  // 交叉熵损失函数
        optimizer: tf.train.adam(0.1),  // 优化器
        metrics: ['accuracy']  // 准确度度量
    })

    await model.fit(xTrain, yTrain, {
        epochs: 100,
        validationData: [xTest, yTest],  // 引入验证集
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss', 'val_loss', 'acc', 'val_acc'],  // 可视化训练集和验证集损失函数和训练和验证集准确度
            { callbacks: ['onEpochEnd'] }  // 设置只显示onEpochEnd
        )
    });

    window.predict = (form) => {
        const input = tf.tensor([[
            form.a.value * 1,
            form.b.value * 1,
            form.c.value * 1,
            form.d.value * 1
        ]]);
        const pred = model.predict(input);
        alert(`预测结果：${IRIS_CLASSES[pred.argMax(1).dataSync(0)]}`); // 输出第二维即训练集中标签的最大概率并转换数据结构
    };


};