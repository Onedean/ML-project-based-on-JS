import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

window.onload = async () => {
    //准备、可视化训练数据
    const xs = [1, 2, 3, 4];
    const ys = [1, 3, 5, 7];

    tfvis.render.scatterplot(
        { name: '线性回归训练集' },
        { values: xs.map((x, i) => ({ x, y: ys[i] })) },
        { xAxisDomain: [0, 5], yAxisDomain: [0, 8] }
    );

    //定义模型结构
    const model = tf.sequential();  // 初始化一个连续的神经网络模型
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));  // 为神经网络模型添加全连接层，其中神经元个数位1，tensor形状为1维
    model.compile({ loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.1) });  // 设置损失函数（采用均方误差MSE），设置sgd优化器（学习速率超参数为0.1）

    // 训练模型并可视化操作过程
    const inputs = tf.tensor(xs);  // 样本
    const labels = tf.tensor(ys);  // 标签
    await model.fit(inputs, labels, {  // 拟合
        batchSize: 4,  // 小批量处理超参数
        epochs: 400,  // 处理次数
        callbacks: tfvis.show.fitCallbacks(  // 可视化
            { name: '训练过程' },
            ['loss']  // 损失函数可视化
        )
    })

    // 进行预测
    const output = model.predict(tf.tensor([5]));  // 将待测数据转化为tensor，使用训练好的模型预测
    alert(`如果 x 为 5, 那么 y 为 ${output.dataSync()[0]}`);  // 将输出的tensor转化为普通数据显示
}